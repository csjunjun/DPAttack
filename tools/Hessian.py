#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*
# firstly you have to git clone https://github.com/amirgholami/PyHessian.git
#then put this file under the downloaded folder
from __future__ import print_function
from PIL import Image
import json,pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import  transforms

from utils import *
from pyhessian import hessian
import torchvision.models as models
# Settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument(
    '--mini-hessian-batch-size',
    type=int,
    default=1,
    help='input batch size for mini-hessian batch (default: 200)')
parser.add_argument('--hessian-batch-size',
                    type=int,
                    default=1,
                    help='input batch size for hessian (default: 200)')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    help='random seed (default: 1)')
parser.add_argument('--batch-norm',
                    action='store_false',
                    help='do we need batch norm or not')
parser.add_argument('--residual',
                    action='store_false',
                    help='do we need residual connect or not')

parser.add_argument('--cuda',
                    action='store_false',
                    help='do we use gpu or not')
parser.add_argument('--resume',
                    type=str,
                    default='',
                    help='get the checkpoint')
def mysortkey(filename:str):
    return int(filename.split(".")[0])  

def mysortkey2(filename:str):
    return int(filename.split("_")[-1].split(".")[0])  

def mysortkey_first(filename:str):
    return int(filename.split("_")[0])
args = parser.parse_args()
# set random seed to reproduce the work
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))

##############
# Get the hessian data
##############
assert (args.hessian_batch_size % args.mini_hessian_batch_size == 0)
assert (50000 % args.hessian_batch_size == 0)
batch_num = args.hessian_batch_size // args.mini_hessian_batch_size

model = models.resnet50(pretrained=True)
if args.cuda:
    model = model.cuda()
model = torch.nn.DataParallel(model)
model.eval()
criterion = nn.CrossEntropyLoss()  # label loss
_mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
_std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
anf = "../data/savefolder_forintermediateAEsofmultipleImages"
print(f"analyze files:{anf}")
saveftmp = f'{anf}_maxeigen'
if not os.path.exists(saveftmp):
    os.mkdir(saveftmp)
subdir = os.listdir(anf)
subdir.sort(key=mysortkey2)
trans = transforms.Compose([
    transforms.Resize((224,224)),
        transforms.ToTensor()
        ])
cleanimg_base = '../data/imagenet/val'
outputdir = "../data/hessianoutput"
ground_truth  = open(os.path.join('../data/imagenet/imagenet_test.txt'), 'r').read().split('\n')

if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    print(f"create dir:{outputdir}")

top_eigenvalues_max =[]
outi = 0
for sub in subdir:#sub: a folder named with the filename of each clean image. The folder sub contains the intermediate AEs of this clean image
    advimglist = [f for f in os.listdir(f'{anf}/{sub}') if f.endswith('.npy')] 
    advimglist.sort(key=mysortkey_first)
    ttt = 0
    ground_name=""
    while ground_name!=sub:
        ground_name_label = ground_truth[ttt]
        
        ground_label_split_all =  ground_name_label.split
        
        ground_label_split =  ground_name_label.split()
        
        ground_label =  ground_name_label.split()[1]
        ground_name =  ground_name_label.split()[0]
        ttt+=1 


    ground_label = torch.tensor([int(ground_label)]).cuda()
    print(f"ground_label:{int(ground_label)}")
    top_eigenvalues_dict = {}
    top_eigenvalues_prev = [0]
    prev_q = 0

    clean_image = Image.open(os.path.join(cleanimg_base,sub)).convert("RGB")
    clean_image = trans(clean_image).cuda().unsqueeze(0)
    clean_image_normed = (clean_image-_mean_torch)/_std_torch

    with torch.no_grad():
        output = model(clean_image_normed)
        
    pred = torch.argmax(output, dim=1)
    hessian_comp = hessian(model,
                        criterion,
                        data=(clean_image_normed, pred),
                        cuda=args.cuda) 
    top_eigenvalues, _ = hessian_comp.eigenvalues()

    top_eigenvalues_max.append([float(top_eigenvalues[0])])
    i=1
    for advimgp in advimglist:
        indexi = int(advimgp.split('_')[0])
        r = float(advimgp.split('_')[1].split("rb")[1])
        query = advimgp.split('_')[0]
        advimg = np.load(f'{anf}/{sub}/'+advimgp)
        
        advimg = torch.from_numpy(advimg).float()
        advimg = advimg.cuda()
        advimg_normed = (advimg-_mean_torch)/_std_torch

        linf = torch.norm((clean_image-advimg).view(-1),p=np.inf)
    
        with torch.no_grad():
            output = model(advimg_normed)
            
        pred = torch.argmax(output, dim=1)
        hessian_comp = hessian(model,
                            criterion,
                            data=(advimg_normed, pred),
                            cuda=args.cuda) 
        top_eigenvalues, _ = hessian_comp.eigenvalues()

        print(f'top_eigenvalues:{float(top_eigenvalues[0])},rbest:{r}' )
        if indexi>=i:
            for ki in range(i,indexi+1):
                top_eigenvalues_max[-1].append(float(top_eigenvalues[0]))
        i = indexi+1
    tracelen = len(top_eigenvalues_max[-1])
    lastitem = top_eigenvalues_max[-1][-1]
    for ki in range(tracelen,100):
        top_eigenvalues_max[-1].append(lastitem)

    outi+=1
savef = f'{anf}_maxeigen/{sub.split(".")[0]}.pkl'
print(f"save to {savef}")
with open(savef, 'wb') as f:
    pickle.dump(top_eigenvalues_max, f)

