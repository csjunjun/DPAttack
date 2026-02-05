# coding:utf-8
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
import math
import argparse
import torch,torchvision
import numpy as np
import copy
import csv
import random
from torchvision import transforms
import sys
from datetime import datetime
from torchvision.utils import save_image

from torchvision import models

from ..tools.DataTools import ADBEvaluate

from .ADBAClass import Block,V
    
##################################################################################################
#represent blocks of a picture


#represent Iterations
class Iter:
    def __init__(self, init_vbest, offspringN, iter_n=1,early_stop=False,tracker=None,PARA_TYPE=9,get_iou_auto=None\
                 ,mask_gt=None,p=None,iouThreshold=0.5):
        self.offspringN = offspringN
        self.iter_n = iter_n
        self.offspringVs = [] #d1 d2
        self.early_stop = early_stop
        self.chosen_v = -1 #
        self.old_vbest = copy.deepcopy(init_vbest) # dbest
        for i in range(offspringN):
            self.offspringVs.append(copy.deepcopy(init_vbest))
            self.offspringVs[i].Rmax, self.offspringVs[i].Rmin = 1.0, 0.0
            #
            #
            #
        self.blacklight_threshold = 25 
        self.blacklight_count = 0
        self.blacklight_first_detect = 0
        self.tracker = tracker
        self.adbafun = ADBEvaluate(PARA_TYPE=PARA_TYPE)
        self.miniou = 1.0
        self.iouThreshold =iouThreshold
        self.get_iou_auto = get_iou_auto 
        self.mask_gt =mask_gt 
        self.p = p
    #create a new generation
    def mutation(self, model, original_image, label, aim_r, tolerance_binary_iters, blocks, \
                 binaryM,method,globalq=0,adaptive=0,gtgrad=None,cossim=[]):
        query = 0
        for vi in range(self.offspringN):
            self.offspringVs[vi].reverse_v(blocks[vi])
            self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = self.old_vbest.Rmax, 0.0

        query_plus,cossim = self.compare_directions_usingADB(
            model, original_image, label, aim_r, tolerance_binary_iters, binaryM,\
                method,globalq,adaptive=adaptive,gtgrad=gtgrad,cossim=cossim)

        query = query + query_plus
        for vi in range(self.offspringN): #initialize directions
            self.offspringVs[vi].reverse_v(blocks[vi])
        if self.chosen_v >= 0:
            self.old_vbest.reverse_v(blocks[self.chosen_v])
            self.old_vbest.Rmax, self.old_vbest.Rmin = (
                self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)
            for vi in range(self.offspringN):
                self.offspringVs[vi].reverse_v(blocks[self.chosen_v])
                self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = (
                    self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)
        self.iter_n = self.iter_n + 1
        return query,cossim
    #

    def predictImg(self,model,image):
        with torch.no_grad():
            logits_clean, _ = model.forward(image, *self.p)
            logits_clean = logits_clean.cuda()
            mask_clean = logits_clean > model.model.mask_threshold
            mask_clean = mask_clean.cuda()
            iou_adv_img = self.get_iou_auto(mask_clean.cpu().detach().numpy(), self.mask_gt)
        return iou_adv_img
    def compare_directions_usingADB(self, model, original_image, label, aim_r, maxIters,\
                                     binaryM,method,globalq=0,paras=None,adaptive=0,gtgrad=None,cossim=[]):
        perturbations = [] #d1 d2
        perturbed_images = [] #= x+ADB*d
        predicted = [] #= F(x+ADB*d)
        query = 0
        succV = []
        self.chosen_v = -1
        #initialize directions
        for i in range(len(self.offspringVs)):
            perturbations.append(self.offspringVs[i].advv_to_tensor().cuda())
            perturbed_images.append(torch.clamp(original_image.cuda() +
                                            self.old_vbest.Rmax * perturbations[i], 0.0, 1.0))
            iou_adv_img = self.predictImg(model,perturbed_images[i])
            predicted.append(iou_adv_img)
            query = query + 1
            if predicted[i] <self.iouThreshold: #success
                succV.append(i)
                self.offspringVs[i].Rmax = self.old_vbest.Rmax
                self.chosen_v = i
                self.miniou = iou_adv_img
            else:
                self.offspringVs[i].Rmin = self.old_vbest.Rmax
        if len(succV) == 0:
            self.chosen_v = -1
            return query,cossim
        elif len(succV) == 1:
            self.chosen_v = succV[0]
            return query,cossim

        low, high = 0, self.old_vbest.Rmax
        for ite in range(0, maxIters): #conmaration loop
            ADB = self.adbafun.next_ADB(low, high, aim_r, self.old_vbest.Rmax, binaryM,method=method,paras=paras)
            succVtemp = copy.deepcopy(succV)
            vi = 0
            while vi < len(succVtemp):
                if adaptive==0:
                    perturbed_images[succVtemp[vi]] = torch.clamp(
                    original_image.cuda() + ADB * perturbations[succVtemp[vi]], 0.0, 1.0)
                elif adaptive==2:
                    perturbed_images[succVtemp[vi]] = torch.clamp(
                    original_image.cuda() + ADB * torch.clamp(perturbations[succVtemp[vi]]+torch.randn_like(original_image[0]).cuda(),-1,1), 0.0, 1.0)


                iou_adv_img = self.predictImg(model,perturbed_images[succVtemp[vi]])

                predicted[succVtemp[vi]] = iou_adv_img
                query = query + 1
                if predicted[succVtemp[vi]] <self.iouThreshold:
                    self.offspringVs[succVtemp[vi]].Rmax = ADB
                    self.chosen_v = succVtemp[vi]
                    self.miniou = iou_adv_img
                    if self.offspringVs[succVtemp[vi]].Rmax <= aim_r:
                        self.chosen_v = succVtemp[vi]
                        if gtgrad is not None:
                            cossim=[float(torch.cosine_similarity(torch.tensor(self.offspringVs[succVtemp[vi]].adv_v).unsqueeze(0),torch.sign(gtgrad).cpu().flatten(start_dim=1)))]
                        return query,cossim
                    vi = vi + 1
                else:
                    self.offspringVs[succVtemp[vi]].Rmin = ADB
                    succVtemp.pop(vi)

            if len(succVtemp) == 0:
                low = ADB
            elif len(succVtemp) == 1:
                self.chosen_v = succVtemp[0]
                if gtgrad is not None:
                    cossim=[float(torch.cosine_similarity(torch.tensor(self.offspringVs[self.chosen_v].adv_v).unsqueeze(0),torch.sign(gtgrad).cpu().flatten(start_dim=1)))]
                return query,cossim
            elif len(succVtemp) >= 2:
                high = ADB
                succV = succVtemp

            if ite >= 4 and high - low <= 0.0002:
                break

        #d1 and d2 are close, just return d1
        self.chosen_v = succV[0]
        if gtgrad is not None:
            cossim=[float(torch.cosine_similarity(torch.tensor(self.offspringVs[self.chosen_v].adv_v).unsqueeze(0),torch.sign(gtgrad).cpu().flatten(start_dim=1)))]
        return query,cossim

@torch.no_grad()
def ATK_ADBA(model, original_image_x, label_y, sample_index, aim_r, \
             tolerance_binary_iters, budget,get_iou_auto=None\
                 ,mask_gt=None,p=None,iouThreshold=0.5): 

    channels, size_x, size_y = original_image_x.shape[1], original_image_x.shape[2], original_image_x.shape[3]

    method ="ADBA" 
    pix_num = channels * size_x * size_y

 
    v0 = V(channels, size_x, size_y, 1)

    iter_num = 1
    block_iter = 0
    b0 = Block(0, pix_num - 1)
    bs1 = b0.cut_block(2)
    blocks = [bs1]
    gtgrad = None
    query = 0
    Rline = [[0, 1.0]]
    offspringN = 2
    ITERATION = Iter(v0, offspringN, 1,early_stop=1,tracker=None,PARA_TYPE=15,\
                     get_iou_auto=get_iou_auto,mask_gt=mask_gt,p=p,iouThreshold=iouThreshold)
    
    queryplus,cossim = ITERATION.mutation(model, original_image_x, label_y, aim_r, tolerance_binary_iters, blocks[0],\
                                       1,method=method,globalq=query,adaptive=0,gtgrad=None,\
                                        cossim=[])
    if len(cossim)>0:
        cossimlist = [[cossim[0],queryplus]]
    else:
        cossimlist = []
    query = query + queryplus

    Rline.append([query, ITERATION.old_vbest.Rmax])
    """"""

        
    while (query < budget) and (ITERATION.old_vbest.Rmax > aim_r):  
        block_iter = block_iter + 1
        blocks_i = []
        for i, bi in enumerate(blocks[block_iter - 1]):
            blocks_i.extend(bi.cut_block(offspringN))
            query_plus,cossim = ITERATION.mutation(model, original_image_x, label_y, aim_r, tolerance_binary_iters,
                                            blocks_i[offspringN * i:offspringN * (i + 1)], 1\
                                                ,method=method,globalq=query,adaptive=0,gtgrad=gtgrad,cossim=cossim)
            query = query + query_plus
            if len(cossim)>0:
                cossimlist.append([cossim[0],query])

            Rline.append([query, ITERATION.old_vbest.Rmax])
            iter_num = iter_num + 1
            if (ITERATION.old_vbest.Rmax <= aim_r) or query >= budget:
                break
        blocks.append(copy.deepcopy(blocks_i))

    Rbest = ITERATION.old_vbest.Rmax
    adversarial_v = ITERATION.old_vbest.advv_to_tensor()
    adversarial_image = original_image_x + Rbest * adversarial_v.cuda()
    adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)

    success = 1
    if Rbest > aim_r:
        success = -1
    # nparray = Rbest*np.array(iter_now.vbest.adv_v).flatten()
    adv_img = adversarial_image - original_image_x

    if query >= 1:
        #save and output images and atk images
        Filestring = ("Img"+str(sample_index)+
                     "_Que"+str(query)+
                      "_Time"+str(datetime.now().strftime("%H-%M-%S"))
                      )
        
        # combined_file = DataTools.save_images(original_image_x,
        #                                       adversarial_image,
        #                                       0.5 * ITERATION.old_vbest.Rmax * (1 + adversarial_v),
        #                                       Filestring)
        
    nparray = np.array(adv_img.cpu()).flatten()
    return success, query, ITERATION.iter_n, Rbest, np.linalg.norm(nparray, ord=2),adversarial_image,ITERATION.miniou


def RlineQ(Rline, radius_line, budget):
    start = 0
    for t in range(len(Rline) - 1):
        for q in range(start, min(Rline[t + 1][0], budget)):
            radius_line[q] = radius_line[q] + Rline[t][1]
            start = Rline[t + 1][0]
    return

def mysortkey(filename:str):
    return int(filename.split("_")[2].split(".")[0])  
