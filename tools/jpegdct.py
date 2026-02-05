
import cv2,math
import torch,os
from PIL import Image
import torch.nn as nn
import itertools
import torchvision.transforms as transforms

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 18})
import pandas as pd  
import seaborn as sns
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

y_table = nn.Parameter(torch.from_numpy(y_table))
#
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = nn.Parameter(torch.from_numpy(c_table))


def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x))**3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality*2
    return quality / 100.


class y_dequantize(nn.Module):
    """ Dequantize Y channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width

    """

    def __init__(self, factor=1):
        super(y_dequantize, self).__init__()
        self.y_table = y_table
        self.factor = factor

    def forward(self, image):
        return image * (self.y_table * self.factor)


class c_dequantize(nn.Module):
    """ Dequantize CbCr channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width

    """

    def __init__(self, factor=1):
        super(c_dequantize, self).__init__()
        self.factor = factor
        self.c_table = c_table

    def forward(self, image):
        return image * (self.c_table * self.factor)


class idct_8x8(nn.Module):
    """ Inverse discrete Cosine Transformation
    Input:
        dcp(tensor): batch x height x width
    Output:
        image(tensor): batch x height x width
    """

    # def __init__(self):
    #     super(idct_8x8, self).__init__()
    #     alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    #     self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
    #     tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    #     for x, y, u, v in itertools.product(range(8), repeat=4):
    #         tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
    #             (2 * v + 1) * y * np.pi / 16)
    #     self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def __init__(self,blocksize=8):
        super(idct_8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * (blocksize-1))
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((blocksize, blocksize, blocksize, blocksize), dtype=np.float32)
        for x, y, u, v in itertools.product(range(blocksize), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / (blocksize*2)) * np.cos(
                (2 * v + 1) * y * np.pi / (blocksize*2))
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())


    def forward(self, image):
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class block_merging(nn.Module):
    """ Merge pathces into image
    Inputs:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self,blocksize=8):
        super(block_merging, self).__init__()
        self.blocksize = blocksize

    def forward(self, patches, height, width):
        k = self.blocksize
        batch_size = patches.shape[0]

        image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class chroma_upsampling(nn.Module):
    """ Upsample chroma layers
    Input:
        y(tensor): y channel image
        cb(tensor): cb channel
        cr(tensor): cr channel
    Ouput:
        image(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(chroma_upsampling, self).__init__()

    def forward(self, y, cb, cr):


        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class ycbcr_to_rgb_jpeg(nn.Module):
    """ Converts YCbCr image to RGB JPEG
    Input:
        image(tensor): batch x height x width x 3
    Outpput:
        result(tensor): batch x 3 x height x width
    """

    def __init__(self):
        super(ycbcr_to_rgb_jpeg, self).__init__()
        #see https://fourcc.org/fccyvrgb.php
        matrix = np.array(
            [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128., -128.]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        # result = torch.from_numpy(result)
        result.view(image.shape)
        return result.permute(0, 3, 1, 2)


class decompress_jpeg(nn.Module):
    """ Full JPEG decompression algortihm
    Input:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        image(tensor): batch x 3 x height x width
    """

    def __init__(self, height=8, width=8, rounding=torch.round, factor=1,bs=8,tonorm=True):
        super(decompress_jpeg, self).__init__()
        self.c_dequantize = c_dequantize(factor=factor)
        self.y_dequantize = y_dequantize(factor=factor)
        self.idct = idct_8x8(blocksize=bs)
        self.merging = block_merging(blocksize=bs)
        self.chroma = chroma_upsampling()
        self.colors = ycbcr_to_rgb_jpeg()
        self.tonorm = tonorm
        self.height, self.width = height, width
        self.bs = bs

    def forward(self, y, cb, cr,height=None,width=None):
        if height is None:
            height = self.height 
        if width is None:
            width = self.width
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():

            comp = self.idct(components[k])
            components[k] = self.merging(comp, height, width)
            #
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)
        if self.tonorm:

            image = torch.min(255 * torch.ones_like(image),
                        torch.max(torch.zeros_like(image), image))
            return image / 255
        else:
            return image / 255

        


class rgb_to_ycbcr_jpeg(nn.Module):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(rgb_to_ycbcr_jpeg, self).__init__()
        matrix = np.array(
            [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0., 128., 128.]))
        #
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        #    result = torch.from_numpy(result)
        result.view(image.shape)
        return result


class chroma_subsampling(nn.Module):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """

    def __init__(self):
        super(chroma_subsampling, self).__init__()

    def forward(self, image):

        return image[:, :, :, 0], image[:, :, :, 1], image[:, :, :, 2]


class block_splitting(nn.Module):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output:
        patch(tensor):  batch x h*w/64 x h x w
    """

    def __init__(self,k=8):
        super(block_splitting, self).__init__()
        self.k = k

    def forward(self, image):
        height, width = image.shape[1:3]
        batch_size = image.shape[0]

        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)


class dct_8x8(nn.Module):
    """ Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """


    def __init__(self,blocksize=8):
        super(dct_8x8,self).__init__()
        tensor = np.zeros((blocksize, blocksize, blocksize, blocksize), dtype=np.float32)
        for x, y, u, v in itertools.product(range(blocksize), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / (2*blocksize)) * np.cos(
                (2 * y + 1) * v * np.pi / (2*blocksize))
        alpha = np.array([1. / np.sqrt(2)] + [1] * (blocksize-1))
        #
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

    def forward(self, image):
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class y_quantize(nn.Module):
    """ JPEG Quantization for Y channel
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self, rounding, factor=1):
        super(y_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.y_table = y_table

    def forward(self, image):
        image = image.float() / (self.y_table * self.factor)
        image = self.rounding(image)
        return image


class c_quantize(nn.Module):
    """ JPEG Quantization for CrCb channels
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self, rounding, factor=1):
        super(c_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.c_table = c_table

    def forward(self, image):
        image = image.float() / (self.c_table * self.factor)
        image = self.rounding(image)
        return image


class compress_jpeg(nn.Module):
    """ Full JPEG compression algortihm
    Input:
        imgs(tensor): batch x 3 x height x width
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
    """

    def __init__(self, rounding=torch.round, factor=1,bs=8):
        super(compress_jpeg, self).__init__()
        self.l1 = nn.Sequential(
            rgb_to_ycbcr_jpeg(),
            chroma_subsampling()
        )
        self.l2 = nn.Sequential(
            block_splitting(k=bs),
            dct_8x8(blocksize=bs)
        )
        self.c_quantize = c_quantize(rounding=rounding, factor=factor)
        self.y_quantize = y_quantize(rounding=rounding, factor=factor)

    def forward(self, image):
        y, cb, cr = self.l1(image * 255)

        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            # if k in ('cb', 'cr'):
            #     comp = self.c_quantize(comp)
            # else:
            #     comp = self.y_quantize(comp)

            components[k] = comp

        return components['y'], components['cb'], components['cr']


class DiffJPEG(nn.Module):
    def __init__(self, differentiable=True, quality=80,bs=8,tonorm=True):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor,bs=bs)
        self.decompress = decompress_jpeg(rounding=rounding, factor=factor,bs=bs,tonorm=tonorm)
        
    def rec(self,y,cb,cr,height,width):
        return self.decompress(y,cb,cr,height,width)
    def forward(self, x,xfilter=None,ycbcr=True,forged=True,batch=False):
        
        if ycbcr:
            y, cb, cr = self.compress(x)
        else:
            bs = block_splitting()
            y = bs(x[0][0].unsqueeze(0))
            cb = bs(x[0][1].unsqueeze(0))
            cr = bs(x[0][2].unsqueeze(0))
        
        if forged:
            y_pristine,cb_pristine,cr_pristine = None,None,None 
            y_forged,cb_forged,cr_forged = None,None,None 
            y_edge,cb_edge,cr_edge = None,None,None
            self.forgidx,self.edgeidx,self.priidx = [],[],[]
            if batch:
                for yi in range(0,y.shape[1]):
                    maskblock = xfilter[0][yi]
                    if torch.all(maskblock==0):#pristine part
                        if y_pristine is None:
                            y_pristine = y[:,yi].unsqueeze(1)
                            cb_pristine = cb[:,yi].unsqueeze(1)
                            cr_pristine = cr[:,yi].unsqueeze(1)
                        else:
                            y_pristine = torch.cat([y_pristine,y[:,yi].unsqueeze(1)],dim=1)
                            cb_pristine = torch.cat([cb_pristine,cb[:,yi].unsqueeze(1)],dim=1)
                            cr_pristine = torch.cat([cr_pristine,cr[:,yi].unsqueeze(1)],dim=1)
                        self.priidx.append(yi)
                    elif torch.all(maskblock==1):#forged part
                        if y_forged is None:
                            y_forged = y[:,yi].unsqueeze(1)
                            cb_forged = cb[:,yi].unsqueeze(1)
                            cr_forged = cr[:,yi].unsqueeze(1)
                        else:
                            y_forged = torch.cat([y_forged,y[:,yi].unsqueeze(1)],dim=1)
                            cb_forged = torch.cat([cb_forged,cb[:,yi].unsqueeze(1)],dim=1)
                            cr_forged = torch.cat([cr_forged,cr[:,yi].unsqueeze(1)],dim=1)
                        self.forgidx.append(yi)
                    else:
                        if y_edge is None:
                            y_edge = y[:,yi].unsqueeze(1)
                            cb_edge = cb[:,yi].unsqueeze(1)
                            cr_edge = cr[:,yi].unsqueeze(1)
                        else:
                            y_edge = torch.cat([y_edge,y[:,yi].unsqueeze(1)],dim=1)
                            cb_edge = torch.cat([cb_edge,cb[:,yi].unsqueeze(1)],dim=1)
                            cr_edge = torch.cat([cr_edge,cr[:,yi].unsqueeze(1)],dim=1)
                        self.edgeidx.append(yi)

                return y_pristine,cb_pristine,cr_pristine,y_forged,cb_forged,cr_forged,y_edge,cb_edge,cr_edge
            else:
                for yi in range(0,y.shape[1]):
                    maskblock = xfilter[0][yi]
                    if torch.all(maskblock==0):#pristine part
                        if y_pristine is None:
                            y_pristine = y[0][yi].unsqueeze(0)
                            cb_pristine = cb[0][yi].unsqueeze(0)
                            cr_pristine = cr[0][yi].unsqueeze(0)
                        else:
                            y_pristine = torch.cat([y_pristine,y[0][yi].unsqueeze(0)],dim=0)
                            cb_pristine = torch.cat([cb_pristine,cb[0][yi].unsqueeze(0)],dim=0)
                            cr_pristine = torch.cat([cr_pristine,cr[0][yi].unsqueeze(0)],dim=0)
                        self.priidx.append(yi)
                    elif torch.all(maskblock==1):#forged part
                        if y_forged is None:
                            y_forged = y[0][yi].unsqueeze(0)
                            cb_forged = cb[0][yi].unsqueeze(0)
                            cr_forged = cr[0][yi].unsqueeze(0)
                        else:
                            y_forged = torch.cat([y_forged,y[0][yi].unsqueeze(0)],dim=0)
                            cb_forged = torch.cat([cb_forged,cb[0][yi].unsqueeze(0)],dim=0)
                            cr_forged = torch.cat([cr_forged,cr[0][yi].unsqueeze(0)],dim=0)
                        self.forgidx.append(yi)
                    else:
                        if y_edge is None:
                            y_edge = y[0][yi].unsqueeze(0)
                            cb_edge = cb[0][yi].unsqueeze(0)
                            cr_edge = cr[0][yi].unsqueeze(0)
                        else:
                            y_edge = torch.cat([y_edge,y[0][yi].unsqueeze(0)],dim=0)
                            cb_edge = torch.cat([cb_edge,cb[0][yi].unsqueeze(0)],dim=0)
                            cr_edge = torch.cat([cr_edge,cr[0][yi].unsqueeze(0)],dim=0)
                        self.edgeidx.append(yi)

                return y_pristine,cb_pristine,cr_pristine,y_forged,cb_forged,cr_forged,y_edge,cb_edge,cr_edge
        else:
            if batch:
                return y,cb,cr
            else:
                return y.squeeze(0),cb.squeeze(0),cr.squeeze(0)


def findOrder(matrix,rows=8,columns=8):

	result = [0]*(rows*columns)
	result[0] = matrix[0][0]
	k = 1
	i = j = 0
	while(k < rows*columns):
		while i >= 1 and j < rows-1:
			i -= 1
			j += 1
			result[k] = matrix[i][j]
			k += 1
		if j < rows-1:
			j += 1
			result[k] = matrix[i][j]
			k += 1
		elif i < columns-1:
			i += 1
			result[k] = matrix[i][j]
			k += 1
		while i < columns-1 and j >= 1:
			i += 1
			j -= 1
			result[k] = matrix[i][j]
			k += 1
		if i < columns-1:
			i += 1
			result[k] = matrix[i][j]
			k += 1
		elif j < rows-1:
			j += 1
			result[k] = matrix[i][j]
			k += 1
	return result


def getZigzagMeanStd(y_pristine_adv_diff,analyzetype="Std"):
    zigzaglist =[]
    r,c = y_pristine_adv_diff.shape[1],y_pristine_adv_diff.shape[2]
    for blocki in range(0,y_pristine_adv_diff.shape[0]):
        block_content = y_pristine_adv_diff[blocki]
        zigzaglist.append(findOrder(block_content.cpu().numpy(),rows=r,columns=c))
        
    y_pristine_adv_diff_mean = np.mean(np.array(zigzaglist),axis=0)
    y_pristine_adv_diff_std = np.std(np.array(zigzaglist),axis=0)

    if analyzetype=="Std":
        pass
    elif analyzetype=="Variance":
        y_pristine_adv_diff_std = [i**2 for i in y_pristine_adv_diff_std]
    
    elif analyzetype=="LogVariance":
        y_pristine_adv_diff_std = [math.log(i**2+1) for i in y_pristine_adv_diff_std]
    return y_pristine_adv_diff_mean,y_pristine_adv_diff_std,zigzaglist
