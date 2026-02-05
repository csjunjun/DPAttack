# coding:utf-8
import os 
import math
import argparse
import torch
import numpy as np
import copy,pywt
import csv
import random
from PIL import Image
from torchvision import transforms
import sys
from datetime import datetime
from torchvision.utils import save_image
from  tools.jpegdct import DiffJPEG
from tools.utils import setSeed,progress_bar
from tools.fetchmodel import fetchImageNetModels
from tools.DataTools import ADBEvaluate
from models.OursClass import Block, V,getZigzagMeanStd,getnewd
import statistics
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns



import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import idct

import os,sys ,json
from torchvision import models


class Iter:
    def __init__(self, init_vbest, offspringN, iter_n=1,early_stop=False,tracker=None,paratype=10,useadba=1):
        self.offspringN = offspringN
        self.iter_n = iter_n
        self.offspringVs = [] #d1 d2
        self.early_stop = early_stop
        self.chosen_v = -1 #chose witch d
        self.old_vbest = copy.deepcopy(init_vbest) # dbest
        for i in range(offspringN):

            self.offspringVs.append(copy.deepcopy(init_vbest))
            self.offspringVs[i].Rmax, self.offspringVs[i].Rmin = 1.0, 0.0
            
        self.blacklight_threshold = 25 
        self.blacklight_count = 0
        self.blacklight_first_detect = 0
        self.tracker = tracker
        self.hessian_batch= 10
        self.hessian_downsample = 4
        self.hessian_mu = 0.001
        self.adbafun = ADBEvaluate(PARA_TYPE=paratype)
        self.adb_interval = 5 
        print(f"self.adb_interval:{self.adb_interval}")
        self.useadba =useadba
    #create a new generation
    def mutation(self, model, original_image, label, aim_r, tolerance_binary_iters, blocks, binaryM,method,globalq=0,hisv=[],hisblock=[],hispre=[],gtgrad=None,nonzeroidx=None):
        query = 0
        self.dim = original_image.shape[1] * original_image.shape[2] * original_image.shape[3]
        cossim=[]
        for vi in range(self.offspringN):
            self.offspringVs[vi].reverse_v(blocks[vi])
            self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = self.old_vbest.Rmax, 0.0
           
        orioffspringlen = len(self.offspringVs)
       
        query_plus,hisv,hisblock,hispre,cossim =  self.compare_directions_usingADB(
            model, original_image, label, aim_r, tolerance_binary_iters, binaryM,method,globalq,hisv,hisblock,blocks,hispre,orioffspringlen,gtgrad=gtgrad,cossim=cossim)

        query = query +query_plus

        for vi in range(orioffspringlen): #initialize directions
            self.offspringVs[vi].reverse_v(blocks[vi])
        if self.chosen_v >= 0:
            if self.chosen_v == orioffspringlen:
                self.old_vbest.adv_v = list(self.offspringVs[self.chosen_v].adv_v)
                self.old_vbest.Rmax, self.old_vbest.Rmin = (
                self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)

                for vi in range(orioffspringlen):
                    self.offspringVs[vi].adv_v = list(self.offspringVs[self.chosen_v].adv_v)
                    self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = (
                        self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)
                    
                self.offspringVs.pop(orioffspringlen)
                #print("change oldvbest 1")

            else:
                self.old_vbest.reverse_v(blocks[self.chosen_v])
                self.old_vbest.Rmax, self.old_vbest.Rmin = (
                    self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)
                for vi in range(orioffspringlen):
                    self.offspringVs[vi].reverse_v(blocks[self.chosen_v])
                    self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = (
                        self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)
                
                if len(self.offspringVs)== (orioffspringlen+1):
                    self.offspringVs.pop(orioffspringlen)
                
        self.iter_n = self.iter_n + 1
        return query,hisv,hisblock,hispre,cossim,[]




    def compare_directions_usingADB(self, model, original_image, label, aim_r, maxIters, binaryM,method,globalq=0,hisv=[],hisblock=[],blocks=[],hispre=[],orioffspringlen=2,gtgrad=None,cossim=[]):
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
            predicted.append(model.predict_label(perturbed_images[i]).cpu())

            if self.tracker is not None:
                match_num = self.tracker.add_img(perturbed_images[i][0].detach().cpu().numpy())
                if match_num>self.blacklight_threshold:
                    self.blacklight_count+=1
                    if self.blacklight_first_detect ==0:
                        self.blacklight_first_detect = globalq+query + 1
            query = query + 1
            if predicted[i] != label:
                succV.append(i)
                self.offspringVs[i].Rmax = self.old_vbest.Rmax
                self.chosen_v = i
            else:
                self.offspringVs[i].Rmin = self.old_vbest.Rmax

        if len(succV) == 0:
            self.chosen_v = -1
            return query,hisv,hisblock,hispre,cossim
       
        elif len(succV) == 1:

            if len(hisv)>0:
                succV.append(orioffspringlen)
                perturbations.append(hisv[0].advv_to_tensor().cuda())
                perturbed_images.append(torch.clamp(original_image.cuda() +
                                    self.old_vbest.Rmax * perturbations[-1], 0.0, 1.0))
                predicted.append(hispre[0])
                self.offspringVs.append(hisv[0])

                hisv.pop(0)
                hispre.pop(0)
            else:
                hisv.append(copy.deepcopy(self.offspringVs[succV[0]]))
                self.chosen_v = succV[0]
                hispre.append(predicted[succV[0]])
                return query,hisv,hisblock,hispre,cossim
       

        low, high = 0, self.old_vbest.Rmax
        for ite in range(0, maxIters): #conmaration loop
            if self.useadba==1:
                ADB = self.adbafun.next_ADB(low, high, aim_r, self.old_vbest.Rmax, binaryM,method=method)
            else:
                ADB = high-(high-low)/self.adb_interval
            succVtemp = copy.deepcopy(succV)
            vi = 0
            while vi < len(succVtemp):
                perturbed_images[succVtemp[vi]] = torch.clamp(
                    original_image.cuda() + ADB * perturbations[succVtemp[vi]], 0.0, 1.0)
                predicted[succVtemp[vi]] = model.predict_label(perturbed_images[succVtemp[vi]]).cpu()
                if self.tracker is not None:
                    match_num = self.tracker.add_img(perturbed_images[succVtemp[vi]][0].detach().cpu().numpy())
                    if match_num>self.blacklight_threshold:
                        self.blacklight_count+=1
                        if self.blacklight_first_detect ==0:
                            self.blacklight_first_detect = globalq+query + 1
                query = query + 1
                if predicted[succVtemp[vi]] != label:
                    self.offspringVs[succVtemp[vi]].Rmax = ADB
                    self.chosen_v = succVtemp[vi]
                    if self.offspringVs[succVtemp[vi]].Rmax <= aim_r:
                        self.chosen_v = succVtemp[vi]
                        if gtgrad is not None:
                            cossim=[float(torch.cosine_similarity(torch.tensor(self.offspringVs[succVtemp[vi]].adv_v).unsqueeze(0),torch.sign(gtgrad).cpu().flatten(start_dim=1)))]
                        return query,hisv,hisblock,hispre,cossim
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
                return query,hisv,hisblock,hispre,cossim
            elif len(succVtemp) >= 2:
                high = ADB
                succV = succVtemp
            if ite >= 4 and high - low <= 0.0002:
                break

        self.chosen_v = succV[0] 
        if gtgrad is not None:
            cossim=[float(torch.cosine_similarity(torch.tensor(self.offspringVs[self.chosen_v].adv_v).unsqueeze(0),torch.sign(gtgrad).cpu().flatten(start_dim=1)))]

        return query,hisv,hisblock,hispre,cossim

@torch.no_grad()
def ATK_ADBA(filename,model, original_image_x, img_number, label_y, sample_index, aim_r, tolerance_binary_iters,\
              order,tracker,initvariables,savep,adbafun,args):
    channels, size_x, size_y = original_image_x.shape[1], original_image_x.shape[2], original_image_x.shape[3]
    if args.channels == 1:
        channels = args.channels
    gtgrad=None
    nonzeroidx=None
    cossimlist_init=None
    method =  ""
    pix_num = channels * size_x * size_y
    npop = 1 
    nchannel = 3
    step_p = args.stepp

    blocksize=args.blocksize
    print("blocksize:",blocksize)
    diffj = DiffJPEG(bs=blocksize)
    dwtlevel = int(math.log2(blocksize))
    if dwtlevel>0 and args.lowtype!="dct" and args.onlyone!=1:
        
        dwtblocksize = 1
        
        print(f"dwtblocksize:{dwtblocksize}")

        if dwtlevel==1:
            cas,(cHs,cVs,cDs) = pywt.dwt2(original_image_x,'haar')
            others = [cHs,cVs,cDs]
        elif dwtlevel==2:
            cas_0,(cHs_0,cVs_0,cDs_0) = pywt.dwt2(original_image_x,'haar')
            cas,(cHs,cVs,cDs) = pywt.dwt2(torch.tensor(cas_0),'haar')
            others = [cas_0,cHs_0,cVs_0,cDs_0,cHs,cVs,cDs]
        elif dwtlevel==3:
            cas_0,(cHs_0,cVs_0,cDs_0) = pywt.dwt2(original_image_x,'haar')
            cas_1,(cHs_1,cVs_1,cDs_1) = pywt.dwt2(torch.tensor(cas_0),'haar')
            cas,(cHs,cVs,cDs) = pywt.dwt2(torch.tensor(cas_1),'haar')

            others = [cas_0,cHs_0,cVs_0,cDs_0,cas_1,cHs_1,cVs_1,cDs_1,cHs,cVs,cDs]
        elif dwtlevel==4:
            cas_0,(cHs_0,cVs_0,cDs_0) = pywt.dwt2(original_image_x,'haar')
            cas_1,(cHs_1,cVs_1,cDs_1) = pywt.dwt2(torch.tensor(cas_0),'haar')
            cas_2,(cHs_2,cVs_2,cDs_2) = pywt.dwt2(torch.tensor(cas_1),'haar')
            cas,(cHs,cVs,cDs) = pywt.dwt2(torch.tensor(cas_2),'haar')

            others = [cas_0,cHs_0,cVs_0,cDs_0,cas_1,cHs_1,cVs_1,cDs_1,cas_2,cHs_2,cVs_2,cDs_2,cHs,cVs,cDs]
        elif dwtlevel == 5:
            cas_0,(cHs_0,cVs_0,cDs_0) = pywt.dwt2(original_image_x,'haar')
            cas_1,(cHs_1,cVs_1,cDs_1) = pywt.dwt2(torch.tensor(cas_0),'haar')
            cas_2,(cHs_2,cVs_2,cDs_2) = pywt.dwt2(torch.tensor(cas_1),'haar')
            cas_3,(cHs_3,cVs_3,cDs_3) = pywt.dwt2(torch.tensor(cas_2),'haar')

            cas,(cHs,cVs,cDs) = pywt.dwt2(torch.tensor(cas_3),'haar')

            others = [cas_0,cHs_0,cVs_0,cDs_0,cas_1,cHs_1,cVs_1,cDs_1,cas_2,cHs_2,cVs_2,cDs_2,cas_3,cHs_3,cVs_3,cDs_3,cHs,cVs,cDs]

    stdinitflag=False
    if args.initDir==-1e5:
        if args.binaryAnalyze==2:
            with torch.enable_grad():
                original_image_x.requires_grad = True
                yc,cb,cr = diffj(original_image_x,forged=False,batch=True)
                ycbcr = torch.cat([yc.unsqueeze(1),cb.unsqueeze(1),cr.unsqueeze(1)],dim=1)
        else:
            yc,cb,cr = diffj(original_image_x.detach().cpu(),forged=False,batch=True)
            ycbcr = torch.cat([yc.unsqueeze(1),cb.unsqueeze(1),cr.unsqueeze(1)],dim=1)
        
        if args.onlyone!=1:
            if args.lowtype == "dct":

                n_block = ycbcr.shape[2]
                if blocksize == 4:
                    lowendf = args.dctTrunc #[0,4*4]
                elif blocksize ==8:
                    lowendf = args.dctTrunc #[0,8*8]
                print(f"lowendf:{lowendf}")

                ldct = torch.zeros_like(ycbcr)
                mdct = torch.zeros_like(ycbcr)
                hdct = torch.zeros_like(ycbcr)

                for lowi in range(0,lowendf):
                    idxi,idxj = getzigzagcor(lowi,rows=blocksize,columns=blocksize)
                    ldct[:,:,:,idxi,idxj] = ycbcr[:,:,:,idxi,idxj] 


                lowpassimg = diffj.rec(ldct[:,0],ldct[:,1],ldct[:,2],original_image_x.shape[2],original_image_x.shape[3])
                lowpassimg = torch.clamp(lowpassimg,0,1).cuda()
                perturb_pixel =  lowpassimg-original_image_x.repeat(npop,1,1,1).cuda()
                dis = torch.norm(perturb_pixel,p=np.inf)
                newd_low = perturb_pixel/torch.norm(perturb_pixel)
                zeronum = len(torch.where(torch.sign(perturb_pixel)==0)[0])
                print(f"newd_low, zeronum:{zeronum}")
                if zeronum>0:
                    newd_low[torch.where(newd_low==0)]=1
                    sign_new_low = torch.sign(newd_low)
                    sign_new_low[torch.where(sign_new_low==0)] = 1
                    tmp = np.array(sign_new_low.detach().flatten(start_dim=1).cpu().numpy(),dtype=np.int32)
                else:
                    tmp = np.array(torch.sign(newd_low).detach().flatten(start_dim=1).cpu().numpy(),dtype=np.int32)
                v0_low_list = []
                for adv_v_init_low in tmp:
                    v0_low = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=adv_v_init_low )
                    v0_low_list.append(v0_low)

            elif args.lowtype == "rcolor":
                if dwtlevel>0:
                    hshape,wshape = cas.shape[2],cas.shape[3]
                else:
                    hshape,wshape =original_image_x.shape[2],original_image_x.shape[3]
                    dwtblocksize = args.blocksize
                randomint_tmp = np.random.randint(0,256,(npop,3,hshape//dwtblocksize,wshape//dwtblocksize) )
                    
                randomint = torch.tensor(randomint_tmp).unsqueeze(4).unsqueeze(5).repeat(1,1,1,1,dwtblocksize,dwtblocksize)
                randomint = randomint.permute(0,1,2,4,3,5).flatten(start_dim=-2)
                randomint = randomint.permute(0,1,4,2,3).flatten(start_dim=-2)
                randomint = randomint/255


                todonp = randomint.detach().cpu().numpy() 
                


            elif args.lowtype=="bar":

                newd = getNewdRays(blocksize=blocksize,h=original_image_x.shape[2],w=original_image_x.shape[3])
                adv_v_init = list(np.array(torch.sign(newd[0]).flatten().cpu().numpy(),dtype=np.int32))
                v0 = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=adv_v_init )
                v0_low_list=[v0]
                todonp=(newd+1)/2
                if dwtlevel==1:
                    cas_pert,(cHs_pert,cVs_pert,cDs_pert) = pywt.dwt2(todonp,'haar')
                    others_pert = [cHs_pert,cVs_pert,cDs_pert]
                elif dwtlevel==2:
                    cas_0_pert,(cHs_0_pert,cVs_0_pert,cDs_0_pert) = pywt.dwt2(todonp,'haar')
                    cas_pert,(cHs_pert,cVs_pert,cDs_pert) = pywt.dwt2(torch.tensor(cas_0_pert),'haar')
                    others = [cas_0,cHs_0,cVs_0,cDs_0,cHs,cVs,cDs]

                cas = torch.tensor(cas)
                todonp = (cas-torch.min(cas))/(torch.max(cas)-torch.min(cas)).cpu().numpy()
            elif args.lowtype=="dwtstd":
                if dwtlevel>0:
                    hshape,wshape = cas.shape[2],cas.shape[3]
                else:
                    hshape,wshape =original_image_x.shape[2],original_image_x.shape[3]
                    dwtblocksize = args.blocksize

                n_block = ycbcr.shape[2]
                y_mu,y_std,_ = getZigzagMeanStd(yc[0])
                cb_mu,cb_std,_ = getZigzagMeanStd(cb[0])
                cr_mu,cr_std,_ = getZigzagMeanStd(cr[0])
                stds =[y_std,cb_std,cr_std]
                mus = [y_mu,cb_mu,cr_mu]
                newd,newcandi,newdis,newd_color,newcandi_color = getNewd(ycbcr,n_block,stds,original_image_x.cuda(),npop=npop,nchannel=nchannel,\
                                            step_p=step_p,diffj=diffj,ord=order,initmu=args.mu,initystd=args.initystd,\
                                                initcbstd=args.initcbstd,initcrstd=args.initcrstd,blocksize=args.blocksize,\
                                                    init=args.init,initvariables=initvariables,freqratio=args.freqratio,color = args.color,returnColorimg=True) 
                todonp = torch.nn.functional.interpolate(newcandi_color,(hshape,wshape)).cpu()
                stdinitflag=True

            if dwtlevel>0 and args.lowtype!="dct":
                cas = torch.tensor(cas)
                global casmin,casmax,casNormalize
                casmin= float(torch.min(cas))
                casmax = float(torch.max(cas))
                casNormalize = (cas-casmin)/(casmax-casmin)

                cHs_repeat = torch.tensor(cHs).repeat(npop,1,1,1).numpy()
                cVs_repeat = torch.tensor(cVs).repeat(npop,1,1,1).numpy()
                cDs_repeat = torch.tensor(cDs).repeat(npop,1,1,1).numpy()
                if  args.replace==0:
                    todonp_unnorm = torch.clamp(casNormalize*0.5 + todonp*0.5,0,1)
                    todonp_unnorm = todonp_unnorm*(casmax-casmin)+casmin
                else:
                    todonp_unnorm = todonp*(casmax-casmin)+casmin


                if dwtlevel==1:
                
                    recons = pywt.idwt2((todonp_unnorm,(np.zeros_like(cHs_repeat),np.zeros_like(cVs_repeat),np.zeros_like(cDs_repeat))),'haar')

                elif dwtlevel==2:
                    cHs_0_repeat = torch.tensor(cHs_0).repeat(npop,1,1,1).numpy()
                    cVs_0_repeat = torch.tensor(cVs_0).repeat(npop,1,1,1).numpy()
                    cDs_0_repeat = torch.tensor(cDs_0).repeat(npop,1,1,1).numpy()

                    recons_1 = pywt.idwt2((todonp_unnorm,(np.zeros_like(cHs_repeat),np.zeros_like(cVs_repeat),np.zeros_like(cDs_repeat))),'haar')
                    recons = pywt.idwt2((recons_1,(np.zeros_like(cHs_0_repeat),np.zeros_like(cVs_0_repeat),np.zeros_like(cDs_0_repeat))),'haar')

                elif dwtlevel == 3:
                    cHs_0_repeat = torch.tensor(cHs_0).repeat(npop,1,1,1).numpy()
                    cVs_0_repeat = torch.tensor(cVs_0).repeat(npop,1,1,1).numpy()
                    cDs_0_repeat = torch.tensor(cDs_0).repeat(npop,1,1,1).numpy()

                    cHs_1_repeat = torch.tensor(cHs_1).repeat(npop,1,1,1).numpy()
                    cVs_1_repeat = torch.tensor(cVs_1).repeat(npop,1,1,1).numpy()
                    cDs_1_repeat = torch.tensor(cDs_1).repeat(npop,1,1,1).numpy()

                    recons_1 = pywt.idwt2((todonp_unnorm,(np.zeros_like(cHs_repeat),np.zeros_like(cVs_repeat),np.zeros_like(cDs_repeat))),'haar')
                    recons_0 = pywt.idwt2((recons_1,(np.zeros_like(cHs_1_repeat),np.zeros_like(cVs_1_repeat),np.zeros_like(cDs_1_repeat))),'haar')
                    recons = pywt.idwt2((recons_0,(np.zeros_like(cHs_0_repeat),np.zeros_like(cVs_0_repeat),np.zeros_like(cDs_0_repeat))),'haar')

                elif dwtlevel == 4:
                    cHs_0_repeat = torch.tensor(cHs_0).repeat(npop,1,1,1).numpy()
                    cVs_0_repeat = torch.tensor(cVs_0).repeat(npop,1,1,1).numpy()
                    cDs_0_repeat = torch.tensor(cDs_0).repeat(npop,1,1,1).numpy()

                    cHs_1_repeat = torch.tensor(cHs_1).repeat(npop,1,1,1).numpy()
                    cVs_1_repeat = torch.tensor(cVs_1).repeat(npop,1,1,1).numpy()
                    cDs_1_repeat = torch.tensor(cDs_1).repeat(npop,1,1,1).numpy()

                    cHs_2_repeat = torch.tensor(cHs_2).repeat(npop,1,1,1).numpy()
                    cVs_2_repeat = torch.tensor(cVs_2).repeat(npop,1,1,1).numpy()
                    cDs_2_repeat = torch.tensor(cDs_2).repeat(npop,1,1,1).numpy()


                    recons_2 = pywt.idwt2((todonp_unnorm,(np.zeros_like(cHs_repeat),np.zeros_like(cVs_repeat),np.zeros_like(cDs_repeat))),'haar')
                    recons_1 = pywt.idwt2((recons_2,(np.zeros_like(cHs_2_repeat),np.zeros_like(cVs_2_repeat),np.zeros_like(cDs_2_repeat))),'haar')
                    recons_0 = pywt.idwt2((recons_1,(np.zeros_like(cHs_1_repeat),np.zeros_like(cVs_1_repeat),np.zeros_like(cDs_1_repeat))),'haar')
                    recons = pywt.idwt2((recons_0,(np.zeros_like(cHs_0_repeat),np.zeros_like(cVs_0_repeat),np.zeros_like(cDs_0_repeat))),'haar')

                elif dwtlevel == 5:

                    cHs_0_repeat = torch.tensor(cHs_0).repeat(npop,1,1,1).numpy()
                    cVs_0_repeat = torch.tensor(cVs_0).repeat(npop,1,1,1).numpy()
                    cDs_0_repeat = torch.tensor(cDs_0).repeat(npop,1,1,1).numpy()

                    cHs_1_repeat = torch.tensor(cHs_1).repeat(npop,1,1,1).numpy()
                    cVs_1_repeat = torch.tensor(cVs_1).repeat(npop,1,1,1).numpy()
                    cDs_1_repeat = torch.tensor(cDs_1).repeat(npop,1,1,1).numpy()

                    cHs_2_repeat = torch.tensor(cHs_2).repeat(npop,1,1,1).numpy()
                    cVs_2_repeat = torch.tensor(cVs_2).repeat(npop,1,1,1).numpy()
                    cDs_2_repeat = torch.tensor(cDs_2).repeat(npop,1,1,1).numpy()


                    cHs_3_repeat = torch.tensor(cHs_3).repeat(npop,1,1,1).numpy()
                    cVs_3_repeat = torch.tensor(cVs_3).repeat(npop,1,1,1).numpy()
                    cDs_3_repeat = torch.tensor(cDs_3).repeat(npop,1,1,1).numpy()

                    recons_3 = pywt.idwt2((todonp_unnorm,(np.zeros_like(cHs_repeat),np.zeros_like(cVs_repeat),np.zeros_like(cDs_repeat))),'haar')
                    recons_2 = pywt.idwt2((recons_3,(np.zeros_like(cHs_3_repeat),np.zeros_like(cVs_3_repeat),np.zeros_like(cDs_3_repeat))),'haar')
                    recons_1 = pywt.idwt2((recons_2,(np.zeros_like(cHs_2_repeat),np.zeros_like(cVs_2_repeat),np.zeros_like(cDs_2_repeat))),'haar')
                    recons_0 = pywt.idwt2((recons_1,(np.zeros_like(cHs_1_repeat),np.zeros_like(cVs_1_repeat),np.zeros_like(cDs_1_repeat))),'haar')
                    recons = pywt.idwt2((recons_0,(np.zeros_like(cHs_0_repeat),np.zeros_like(cVs_0_repeat),np.zeros_like(cDs_0_repeat))),'haar')



                recons = torch.clamp(torch.tensor(recons),0,1).cuda()

                perturb_pixel =  recons-original_image_x.repeat(npop,1,1,1).cuda()
                dis = torch.norm(perturb_pixel,p=np.inf)
                newd_low = perturb_pixel/torch.norm(perturb_pixel)
                zeronum = len(torch.where(torch.sign(perturb_pixel)==0)[0])
                print(f"newd_low, zeronum:{zeronum}")
                if zeronum>0:
                    newd_low[torch.where(newd_low==0)]=1
                    sign_new_low = torch.sign(newd_low)
                    sign_new_low[torch.where(sign_new_low==0)] = 1
                    tmp = np.array(sign_new_low.detach().flatten(start_dim=1).cpu().numpy(),dtype=np.int32)
                else:
                    tmp = np.array(torch.sign(newd_low).detach().flatten(start_dim=1).cpu().numpy(),dtype=np.int32)
                v0_low_list = []
                for adv_v_init_low in tmp:
                    v0_low = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=adv_v_init_low )
                    v0_low_list.append(v0_low)
            
        if args.onlyone<2:
            if args.binaryAnalyze==2  or args.binaryAnalyze==4:
                n_block = ycbcr.shape[2]
                y_mu,y_std,_ = getZigzagMeanStd(yc[0])
                cb_mu,cb_std,_ = getZigzagMeanStd(cb[0])
                cr_mu,cr_std,_ = getZigzagMeanStd(cr[0])
                stds =[y_std,cb_std,cr_std]
                mus = [y_mu,cb_mu,cr_mu]
                newd,newcandi,newdis,newd_color = getNewd(ycbcr,n_block,stds,original_image_x.cuda(),npop=npop,nchannel=nchannel,\
                                            step_p=step_p,diffj=diffj,ord=order,initmu=args.mu,initystd=args.initystd,\
                                                initcbstd=args.initcbstd,initcrstd=args.initcrstd,blocksize=args.blocksize,\
                                                    init=args.init,initvariables=initvariables,freqratio=args.freqratio,color = args.color) 
                
                zeronum = len(torch.where(torch.sign(newd)==0)[0])
                print(f"newd, zeronum:{zeronum}")
                
                if zeronum>0:
                    newd[torch.where(newd==0)]=1
                    sign_new_low = torch.sign(newd)
                    sign_new_low[torch.where(sign_new_low==0)] = 1
                    adv_v_init = list(np.array(sign_new_low[0].flatten().detach().cpu().numpy(),dtype=np.int32))
                else:
                    adv_v_init = list(np.array(torch.sign(newd[0]).flatten().detach().cpu().numpy(),dtype=np.int32))

                v0 = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=adv_v_init )
            else:
                if stdinitflag==False:
                    n_block = ycbcr.shape[2]
                    y_mu,y_std,_ = getZigzagMeanStd(yc[0])
                    #print(list(y_std[0:8]))
                    cb_mu,cb_std,_ = getZigzagMeanStd(cb[0])
                    cr_mu,cr_std,_ = getZigzagMeanStd(cr[0])
                    stds =[y_std,cb_std,cr_std]
                    mus = [y_mu,cb_mu,cr_mu]
                    newd,newcandi,newdis,newd_color = getNewd(ycbcr,n_block,stds,original_image_x.cuda(),npop=npop,nchannel=nchannel,\
                                                step_p=step_p,diffj=diffj,ord=order,initmu=args.mu,initystd=args.initystd,\
                                                    initcbstd=args.initcbstd,initcrstd=args.initcrstd,blocksize=args.blocksize,\
                                                        init=args.init,initvariables=initvariables,freqratio=args.freqratio,color = args.color) 


                zeronum = len(torch.where(torch.sign(newd)==0)[0])
                print(f"newd, zeronum:{zeronum}")
                if zeronum>0:
                    newd[torch.where(newd==0)]=1
                    sign_new_low = torch.sign(newd)
                    sign_new_low[torch.where(sign_new_low==0)] = 1
                    adv_v_init = list(np.array(sign_new_low[0].flatten().detach().cpu().numpy(),dtype=np.int32))
                else:
                    adv_v_init = list(np.array(torch.sign(newd[0]).flatten().detach().cpu().numpy(),dtype=np.int32))

                v0 = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=adv_v_init )
                if args.lowtype=="dct":
                    adv_v_init_color = list(np.array(torch.sign(newd_color[0]).flatten().cpu().numpy(),dtype=np.int32))
                    v0_color = V(channels, size_x, size_y, args.initDir,adv_v=adv_v_init_color )




    else:
        v0 = V(channels, size_x, size_y, args.initDir)

    #init binary search
    if args.onlyone!=1:
        newd_image_low = []
        for v0_low in v0_low_list:
            newd_image_low_tmp = v0_low.advv_to_tensor()
            newd_image_low.append(newd_image_low_tmp)
        newd_image_low = torch.stack(newd_image_low)

    if args.binaryAnalyze==-1:
        gtgrad= torch.tensor(torch.load(f"../GradSignSimilarity/FGSM_CE_origianlx/{filename}.pth"))
    if args.onlyone<=1:
        if args.binaryAnalyze==2  or args.binaryAnalyze==4:
            newd_image = torch.sign(newd)
        elif args.binaryAnalyze==-1 or args.binaryAnalyze==5 or args.binaryAnalyze==6 or args.binaryAnalyze==7:
            stname = f"../GradSignSimilarity/startpoints/rs50_bs4_onlystd/{filename}.npy"
            if not os.path.exists(stname):
                np.save(stname,v0.adv_v)
                newd_image = torch.sign(newd).cuda()
            else:
                tmp = np.load(stname)
                newd_image = torch.tensor(tmp).reshape((1,3,224,224)).cuda()
            if gtgrad is not None:
                cossimlist_init = float(torch.cosine_similarity(torch.sign(newd_image).flatten(start_dim=1).cpu(),torch.sign(gtgrad).flatten(start_dim=1),dim=1))
        else:
            newd_image = v0.advv_to_tensor()
        newd_image_binary = newd_image
        if args.binaryAnalyze==2:
            np.save(f"/home/code/attacks/ADBA/code/startpoints_wrs50/{filename}.npy",v0.adv_v)
    else:
        v0 = v0_low_list[0]
        newd_image_binary = newd_image_low
       
    if args.budget<=20 or args.onlyone>0:
        query = 0
        initrhigh=1
        initrlow=0
        success = -1
        candi = None
        chosenv_list = []
        thislimit = min(10,args.budget) if args.onlyone>0 else min(20,args.budget)
        while query<thislimit  :

            mid = (initrhigh+initrlow)/2
            
            if args.binaryAnalyze==2  or args.binaryAnalyze==4 or args.binaryAnalyze==-1 or args.binaryAnalyze==5 or args.binaryAnalyze==6  or args.binaryAnalyze==7:
                candi = torch.clamp(original_image_x.cuda()+mid*newd_image_binary,0,1)
                pre = torch.argmax(model(candi))
            else:
                candi = torch.clamp(original_image_x+mid*newd_image_binary,0,1)
                pre = torch.argmax(model(candi.cuda())).cpu()
            query+=1
            if args.binaryAnalyze==2  or args.binaryAnalyze==4 or args.binaryAnalyze==-1  or args.binaryAnalyze==5 or args.binaryAnalyze==6  or args.binaryAnalyze==7:
                dis = torch.norm(candi-original_image_x.cuda(),p=np.inf)
            else:
                dis = torch.norm(candi-original_image_x,p=np.inf)
            if pre==label_y:
                initrlow = mid 
            else:
                initrhigh = mid 
                if dis<=aim_r:
                    success = 1
                    break
        args.paratype=22
        if args.binaryAnalyze==2 or args.binaryAnalyze==4 or args.binaryAnalyze==-1 or args.binaryAnalyze==5  or args.binaryAnalyze==6  or args.binaryAnalyze==7:
            tmp = torch.clamp(original_image_x.cuda()+initrhigh*newd_image_binary,0,1)
            dis = torch.norm(tmp-original_image_x.cuda(),p=np.inf)
        else:
            tmp = torch.clamp(original_image_x+initrhigh*newd_image_binary,0,1)
            dis = torch.norm(tmp-original_image_x,p=np.inf)

        if args.binaryAnalyze==1:
            if args.onlyone==2 and args.lowtype=="bar":
                np.save(f"../blocksize2startpoint/{args.victimmodel}/{filename}.npy",tmp.detach().cpu().numpy())
                return True, query, 0, initrhigh, 0, 0, [] ,0,0,chosenv_list
            if args.onlyone==2 and args.lowtype=="rcolor":
                bloc_num = len(v0.continue_subarr)
                #np.save(f"../blocksize4startpoint_dwt/{args.victimmodel}/{filename}.npy",tmp.detach().cpu().numpy())
                np.save(f"../blocksize{args.blocksize}startpoint_rcolor/{args.victimmodel}/{filename}.npy",tmp.detach().cpu().numpy())
                return True, query, 0, initrhigh, 0, 0, [] ,0,0,chosenv_list,bloc_num
            if args.onlyone==1:
                bloc_num = len(v0.continue_subarr)
                np.save(f"../blocksize{args.blocksize}startpoint_std/{args.victimmodel}/{filename}.npy",tmp.detach().cpu().numpy())

                return True, query, 0, initrhigh, 0, 0, [] ,0,0,chosenv_list,[],bloc_num
                # return True, query, 0, initrhigh, 0, 0, [] ,0,0,chosenv_list

        elif args.binaryAnalyze==4:
            celos = torch.nn.CrossEntropyLoss(reduction='none')
            if args.victimmodel=="resnet50":
                gtfolder=f"../GradSignSimilarity/FGSM_CE_origianlx"
            else:
                gtfolder=f"../GradSignSimilarity/FGSM_CE_origianlx_{args.victimmodel}"
            gtpath = f"{gtfolder}/{filename}.pth"
            if not os.path.exists(gtfolder):
                os.makedirs(gtfolder)
            if not os.path.exists(gtpath):
                print(f"gtpath:{gtpath} not exists, calculate grad sign")
                # tmp.requires_grad = True
                with torch.enable_grad():
                    img = original_image_x.detach().cuda()
                    img.requires_grad = True
                    output = model(img)
                    cost=celos(output, label_y.unsqueeze(0).cuda())
                    gtgrad = torch.autograd.grad(cost, img,
                                            retain_graph=False, create_graph=False)[0]
                    print(f"grad sign of {filename}:len(0):{len(torch.where(torch.sign(gtgrad)==0)[0])},len(1):{len(torch.where(torch.sign(gtgrad)==1)[0])},len(-1):{len(torch.where(torch.sign(gtgrad)==-1)[0])}")
                    torch.save(gtgrad.detach().cpu().numpy(),gtpath)
            else:
                gtgrad= torch.tensor(torch.load(gtpath))

            nonzeroidx = gtgrad[0].cpu().numpy().flatten()!=0
            tmp = torch.clamp(original_image_x.cuda()+initrhigh*newd_image_binary,0,1)
            cossimlist_init = float(torch.cosine_similarity(torch.sign(tmp.cpu()-original_image_x.cpu()).flatten(start_dim=1).cpu(),torch.sign(gtgrad).flatten(start_dim=1).cpu(),dim=1))#等于1

    else:
        if args.binaryAnalyze==4:
            celos = torch.nn.CrossEntropyLoss(reduction='none')
            if args.victimmodel=="resnet50":
                gtfolder=f"../GradSignSimilarity/FGSM_CE_origianlx"

            else:
                gtfolder=f"../GradSignSimilarity/FGSM_CE_origianlx_{args.victimmodel}"
            gtpath = f"{gtfolder}/{filename}.pth"
            if not os.path.exists(gtfolder):
                os.makedirs(gtfolder)
            if not os.path.exists(gtpath):
                print(f"gtpath:{gtpath} not exists, calculate grad sign")
                # tmp.requires_grad = True
                with torch.enable_grad():
                    
                    img = original_image_x.detach().cuda()
                    img.requires_grad = True
                    output = model(img)
                    cost=celos(output, label_y.unsqueeze(0).cuda())
                    # Update adversarial images
                    gtgrad = torch.autograd.grad(cost, img,
                                            retain_graph=False, create_graph=False)[0]
                    print(f"grad sign of {filename}:len(0):{len(torch.where(torch.sign(gtgrad)==0)[0])},len(1):{len(torch.where(torch.sign(gtgrad)==1)[0])},len(-1):{len(torch.where(torch.sign(gtgrad)==-1)[0])}")
                    torch.save(gtgrad.detach().cpu().numpy(),gtpath)

            else:
                gtgrad= torch.tensor(torch.load(gtpath))

            nonzeroidx = gtgrad[0].cpu().numpy().flatten()!=0
        

        with torch.no_grad():
            query = 0
            initrhigh,initrhigh_low=1,1
            initrlow,initrlow_low=0,0
            success = -1
            candi = None
            chosenv_list = []
            flag_stop ,flag_low_stop = 0,0
            query_1,query_1_low = 0,0
            dis,dis_low,disbest = initrhigh,initrhigh,initrhigh
            innerqlimit = tolerance_binary_iters
            print(f"innerqlimit:{innerqlimit}")
            innerloop = 0
            stop1,stop2 = False,False
            prev_dis_low,prev_dis =1,1
            while query<innerqlimit*2 and query <args.budget and success==-1 and abs(query_1-query_1_low)<2 and not stop1 and not stop2:
                
                if query_1<innerqlimit and (dis == disbest or  flag_stop<2) :
                    stop1=False
                    if  flag_stop==0:
                        mid = (initrhigh+initrlow)/2
                    else:
                        mid = initrhigh-(initrhigh-initrlow)/5
                    candi = torch.clamp(original_image_x+mid*newd_image.cpu(),0,1)
                    pre = torch.argmax(model(candi.cuda())).cpu()
                    query_1 += 1
                    query += 1
                    dis = torch.norm(candi-original_image_x,p=np.inf)
                    if pre==label_y:
                        flag_stop +=1
                        initrlow = mid
                        tmp = prev_dis
                    else:
                        initrhigh = mid 
                        if dis<disbest:
                            disbest = dis
                        if dis<=aim_r:
                            success=1
                            break
                    prev_dis = dis
                    if pre==label_y:
                        dis = tmp
                else:
                    stop1=True
                if  query_1_low<innerqlimit and (dis_low == disbest or flag_low_stop<2):
                    stop2 = False
                    if  flag_low_stop==0:
                        mid_low = (initrhigh_low+initrlow_low)/2
                    else:
                       
                        mid_low = initrhigh_low-(initrhigh_low-initrlow_low)/5
                    if args.lowtype=="dct":
                        candi_low = torch.clamp(lowpassimg+mid_low*newd_image_color,0,1)
                    elif args.lowtype=="rcolor" or args.lowtype=="bar"  or args.lowtype=="dwtstd":
                        candi_low = torch.clamp(original_image_x+mid_low*newd_image_low.cpu(),0,1)

                    pre_low = torch.argmax(model(candi_low.cuda())).cpu()
                    query_1_low += 1
                    query += 1
                    dis_low = torch.norm(candi_low-original_image_x,p=np.inf)
                    if pre_low==label_y:
                        flag_low_stop +=1
                        initrlow_low = mid_low
                        tmp = prev_dis_low
                    else:
                        initrhigh_low = mid_low
                        if dis_low<disbest:
                            disbest=dis_low
                        if dis_low<=aim_r:
                            success=1
                            break
                    prev_dis_low = dis_low
                    if pre_low == label_y:
                        dis_low = tmp
                else:
                    stop2 = True
                
                innerloop += 1
                if innerloop >= 4 and initrhigh_low - initrlow_low <= 0.0002:
                    print("loop threshold.")
                    break
            tmp = torch.clamp(original_image_x+initrhigh*newd_image.cpu(),0,1)
            if args.lowtype=="dct":
                tmp_low = torch.clamp(lowpassimg+initrhigh_low*newd_image_color,0,1)
            elif args.lowtype=="rcolor" or args.lowtype=="bar" or args.lowtype=="dwtstd":
                tmp_low = torch.clamp(original_image_x+initrhigh_low*newd_image_low,0,1)
            dis = torch.norm(tmp-original_image_x,p=np.inf)
            dis_low = torch.norm(tmp_low-original_image_x,p=np.inf)
            #print("#choose the smaller one, equal exclude")
            print(f"dis:{float(dis)},dis_low:{float(dis_low)}")

            if args.binaryAnalyze==11:
                return True, query, 0, initrhigh, 0, 0, [] ,0,0,chosenv_list,(initrhigh_low - initrlow_low)/5,(initrhigh - initrlow)/5
            #candi = tmp
            if dis_low<=dis: 
                candi = tmp_low
                initrhigh = initrhigh_low
                tmpd = candi-original_image_x
                adv_v_init_tmp = list(np.array(torch.sign(tmpd[0]).flatten().cpu().numpy(),dtype=np.int32))
                v0 = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=adv_v_init_tmp )
                args.paratype=22
            else:
                candi = tmp
                args.paratype=22
            if gtgrad is not None:
                cossimlist_init = float(torch.cosine_similarity(torch.sign(candi.cpu()-original_image_x.cpu()).flatten(start_dim=1).cpu(),torch.sign(gtgrad).flatten(start_dim=1).cpu(),dim=1))#等于1
            else:
                cossimlist_init = None
            if args.binaryAnalyze==10:
                direction_save= candi.cpu()-original_image_x.cpu()
                np.save(f"../GradSignSimilarity/startpoints/{args.victimmodel}_bs4_ourscombine/{filename}.npy",direction_save.detach().cpu().numpy())
                return True, query, 0, initrhigh, 0, 0, [] ,0,0,chosenv_list

    print(f"args.paratype:{args.paratype}")
   
    if cossimlist_init is not None:
        cossimlist = [[cossimlist_init,query]]
        cossimavgblock_list=[]
    else:
        cossimlist,cossimavgblock_list = [],[]

    with torch.no_grad():
      
            
        if success == 1 or query >= args.budget:
            Rbest=initrhigh
            adv_img = candi.cuda()-original_image_x.cuda()
            Rline = [[0, 1.0]]
            nparray = np.array(adv_img.cpu()).flatten()
            return success, query, 0, Rbest, np.linalg.norm(nparray, ord=2), np.mean(
                nparray), Rline ,0,0,chosenv_list,cossimlist,cossimavgblock_list # np.linalg.norm(nparray,ord=np.inf)
        elif success==-1 and query <args.budget:
            iter_num = 1
            block_iter = 0

            if args.ablation==0:
                bloc_num = len(v0.continue_subarr)
                b0 = Block(0, bloc_num - 1)
                bs1 = b0.cut_block(args.offspringN)
                blocks = [bs1]
            elif args.ablation==1:
                b0 = Block(0, pix_num - 1)
                bs1 = b0.cut_block(args.offspringN)
                blocks = [bs1]

            
            Rline = [[0, 1.0]]
            v0.Rmax =initrhigh
            ITERATION = Iter(v0, args.offspringN, 1,early_stop=args.early_stop,tracker=tracker,paratype=args.paratype,useadba=args.useadba)
          
            progress_bar(img_number, query, iter_num, ITERATION.old_vbest.Rmax)
            Rline.append([query, ITERATION.old_vbest.Rmax])
            """"""
            singleNext = False
            hisv,hisblock,hispre=[],[],[] 
            blockdir1_stop = False
            while (query < args.budget) and (ITERATION.old_vbest.Rmax > aim_r):  

                block_iter = block_iter + 1
                blocks_i = []
                for i, bi in enumerate(blocks[block_iter - 1]):

                    updated = False
                    if bi.x2-bi.x1==0:
                        if bi.x2==len(ITERATION.old_vbest.continue_subarr):
                            continue
                        print("Block with is 0.")
                        updated_block,updated = ITERATION.old_vbest.split_block(bi)
                        for ijx in range(len(ITERATION.offspringVs)):
                            _,_ = ITERATION.offspringVs[ijx].split_block(bi)
                        if updated:
                            blocks[block_iter - 1][i] = updated_block
                            bi = updated_block
                            for ijx in range(len(blocks[block_iter - 1])):
                                if ijx>i:
                                    blocks[block_iter - 1][ijx].x1 +=1 
                                    blocks[block_iter - 1][ijx].x2 +=1 
                     
                    if updated:
                        assert len(ITERATION.old_vbest.continue_subarr)-1 == blocks[block_iter - 1][ijx].x2

                    blocks_i.extend(bi.cut_block(args.offspringN))
                    
                    query_plus,hisv,hisblock,hispre,cossim,cossimavgblock = ITERATION.mutation(model, original_image_x, label_y, aim_r, tolerance_binary_iters,
                                                    blocks_i[args.offspringN * i:args.offspringN * (i + 1)], args.binaryM,method=method,globalq=query,\
                                                        hisv=hisv,hisblock=hisblock,hispre=hispre,gtgrad=gtgrad,nonzeroidx=nonzeroidx)
                    if len(cossim)>0:
                        cossimlist.append([cossim[0],query])
                    if len(cossimavgblock)>0:
                        cossimavgblock_list.append(cossimavgblock)
                    query = query + query_plus
                    Rline.append([query, ITERATION.old_vbest.Rmax])
                   
                    iter_num = iter_num + 1
                    if (ITERATION.old_vbest.Rmax <= aim_r) or query >= args.budget:
                        break
                if len(blocks_i)>0:
                    blocks.append(copy.deepcopy(blocks_i))
                
            if  (query < args.budget) and (ITERATION.old_vbest.Rmax > aim_r):

                newd_rev,newcandi,newdis,newd_color = getNewd(ycbcr,n_block,stds,original_image_x.cuda(),npop=npop,nchannel=nchannel,\
                                            step_p=step_p,diffj=diffj,ord=order,initmu=args.mu,initystd=args.initystd,\
                                                initcbstd=args.initcbstd,initcrstd=args.initcrstd,blocksize=args.blocksize,\
                                                    init=args.init,initvariables=initvariables,freqratio=args.freqratio,color = args.color) 
                
                zeronum = len(torch.where(torch.sign(newd_rev)==0)[0])
                print(f"newd, zeronum:{zeronum}")
                if zeronum>0:
                    newd_rev[torch.where(newd_rev==0)]=1
                    sign_new_low = torch.sign(newd_rev)
                    sign_new_low[torch.where(sign_new_low==0)] = 1
                    adv_v_init_rev = list(np.array(sign_new_low[0].flatten().detach().cpu().numpy(),dtype=np.int32))
                else:
                    adv_v_init_rev = list(np.array(torch.sign(newd_rev[0]).flatten().detach().cpu().numpy(),dtype=np.int32))

                v0_rev = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=adv_v_init_rev )
                v0_rev.Rmax =ITERATION.old_vbest.Rmax

            innerloop=0
            while  (query < args.budget) and (ITERATION.old_vbest.Rmax > aim_r):
                print(f"innerloop:{innerloop},query:{query},Rbest:{ITERATION.old_vbest.Rmax}")

                block_iter = 0

                if args.ablation==0:
                    bloc_num = len(v0_rev.continue_subarr)
                    b0 = Block(0, bloc_num - 1)
                    bs1 = b0.cut_block(args.offspringN)
                    blocks = [bs1]
                elif args.ablation==1:
                    b0 = Block(0, pix_num - 1)
                    bs1 = b0.cut_block(args.offspringN)
                    blocks = [bs1]

                blockdir2_stop = False
                while (query < args.budget) and (ITERATION.old_vbest.Rmax > aim_r):  

                    block_iter = block_iter + 1
                    blocks_i = []
                    for i, bi in enumerate(blocks[block_iter - 1]):
                        updated = False
                        if bi.x2-bi.x1==0:
                            if bi.x2==len(ITERATION.old_vbest.continue_subarr):
                                #blocks[block_iter - 1].remove(bi)
                                continue
                            print("Seg2 Block with is 0.")
                            updated_block,updated = ITERATION.old_vbest.split_block(bi)
                            for ijx in range(len(ITERATION.offspringVs)):
                                _,_ = ITERATION.offspringVs[ijx].split_block(bi)
                            if updated:
                                blocks[block_iter - 1][i] = updated_block
                                bi = updated_block
                                for ijx in range(len(blocks[block_iter - 1])):
                                    if ijx>i:
                                        blocks[block_iter - 1][ijx].x1 +=1 
                                        blocks[block_iter - 1][ijx].x2 +=1 
                           
                        blocks_i.extend(bi.cut_block(args.offspringN))
                        if updated:
                            assert len(ITERATION.old_vbest.continue_subarr)-1 == blocks[block_iter - 1][ijx].x2

                        query_plus,hisv,hisblock,hispre,cossim,cossimavgblock = ITERATION.mutation(model, original_image_x, label_y, aim_r, tolerance_binary_iters,
                                                        blocks_i[args.offspringN * i:args.offspringN * (i + 1)], args.binaryM,method=method,globalq=query,hisv=hisv,hisblock=hisblock,hispre=hispre)
                        if len(cossimavgblock)>0:
                            cossimlist.append([cossim,query])
                        if len(cossimavgblock)>0:
                            cossimavgblock_list.append(cossimavgblock)
  
                        query = query + query_plus
                        Rline.append([query, ITERATION.old_vbest.Rmax])
                        iter_num = iter_num + 1
                        if (ITERATION.old_vbest.Rmax <= aim_r) or query >= args.budget:
                            break
                    if len(blocks_i)>0:
                        blocks.append(copy.deepcopy(blocks_i))
                  
                innerloop += 1
            Rbest = ITERATION.old_vbest.Rmax
            adversarial_v = ITERATION.old_vbest.advv_to_tensor()
            adversarial_image = original_image_x + Rbest * adversarial_v
            adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)
            

            success = 1
            if Rbest > aim_r:
                success = -1
            if success==1 and args.saveimg==1:           
                save_image(adversarial_image,f'{savep}/{img_number}_{int(label_y)}_{query}.png')

            adv_img = adversarial_image - original_image_x

                
            nparray = np.array(adv_img.cpu()).flatten()
            return success, query, ITERATION.iter_n, Rbest, np.linalg.norm(nparray, ord=2), np.mean(
                nparray), Rline ,ITERATION.blacklight_count,ITERATION.blacklight_first_detect,chosenv_list,cossimlist,cossimavgblock_list # np.linalg.norm(nparray,ord=np.inf)


def RlineQ(Rline, radius_line, budget):
    start = 0
    for t in range(len(Rline) - 1):
        for q in range(start, min(Rline[t + 1][0], budget)):
            radius_line[q] = radius_line[q] + Rline[t][1]
            start = Rline[t + 1][0]
    return

def mysortkey(filename:str):
    return int(filename.split("_")[2].split(".")[0])  

def mysortkey2(filename:str):
    return int(filename.split("_")[2].split("_")[0])  


def main_ADBA():
    # ###################################################################################
    torch_model, test_loader = None, None
    parser = argparse.ArgumentParser(description='Hard Label Attacks')
    parser.add_argument('--victimmodel', default='resnet50', type=str,
                        help='for imagenet:resnet50,vit,inv3,dense121,deit,regnet,SwinV2T,ConvNextBase,efficient,wrs50;for imagenetc:HMany,NoisyMix')
    parser.add_argument('--replace', default=0, type=int,
                        help='Dataset')
    parser.add_argument('--datasource', default='imagenet', type=str,
                        help='imagenet,imagenetc')
    parser.add_argument('--binaryAnalyze', default=0, type=int,
                        help='default:0 to perform attack; others for analysis')

    parser.add_argument('--epsilon', default=0.05, type=float,
                        help='attack strength')
    parser.add_argument('--imgnum', default=200, type=int,
                        help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--beginIMG', default=0, type=int, #43
                        help='begin test img number')

    parser.add_argument('--blocksize', default=4, type=int,
                        help='2,4,8,16,32')  
    parser.add_argument('--dwtlevel', default=2, type=float,
                        help='1,2,...log2d')  
 
    parser.add_argument('--budget', default=100, type=int,
                        help='Maximum queries f0r the attack')
    parser.add_argument('--deviceid', default="1", type=str,
                        help='attack batch size.')
    
    parser.add_argument('--useadba', default= 1, type=int,
                        help='use adba para: 1 or linesearch:0')
    parser.add_argument('--onlyone', default=0, type=int,
                        help='0::DDM,1:use dn;2:low frequency pass direction')
    parser.add_argument('--ablation', default=0, type=int,
                        help='0::PDO,1:dyadic search.')
    parser.add_argument('--lowtype', default="rcolor", type=str,
                        help='default for paper:rcolor;dct,bar,rcolor,dwtstd') 
    parser.add_argument('--dctTrunc', default=1, type=float,
                        help='1 to blocksize**2')  


    
    parser.add_argument('--binaryM', default=1, type=int,
                        help='binary search mod, mid 0 or median 1.')
    parser.add_argument('--early_stop', default=1, type=int,
                        help='early_stop')
    parser.add_argument('--initDir', default=-1e5, type=int,
                        help='initial direction, 1,-1,and 0 for random;-1e5 for our dn init')
    parser.add_argument('--channels', default=3, type=int,
                        help='')
    parser.add_argument('--offspringN', default=2, type=int,
                        help='offspring diretion num in new iteration')
    parser.add_argument('--targeted', default=0, type=int,
                        help='targeted or untargeted')
    parser.add_argument('--norm', default='np.linf', type=str,
                        help='Norm for attack, linf only')
    parser.add_argument('--batch', default=1, type=int,
                        help='attack batch size.')
    parser.add_argument('--early', default='1', type=str,
                        help='early stopping (stop attack once the adversarial example is found)')

    parser.add_argument('--stepp', default=0.01, type=float,
                        help='deprecated')    
    parser.add_argument('--mu', default=0, type=float,
                        help='deprecated')   
    parser.add_argument('--initystd', default=0, type=float,
                        help='deprecated')  
    parser.add_argument('--initcbstd', default=0, type=float,
                        help='deprecated')  
    parser.add_argument('--initcrstd', default=0, type=float,
                        help='deprecated')  

    parser.add_argument('--paratype', default=17, type=int,
                        help='paratype for adba evaluate function')  
    parser.add_argument('--seed', default=0, type=int,
                        help='')  
    parser.add_argument('--saveimg', default=0, type=int,
                        help='')  
    parser.add_argument('--freqratio', default=64, type=int,
                        help='for analysis')  
    parser.add_argument('--color', default=0, type=int,
                        help='default:0 for standard model,1 for robust model')  
    parser.add_argument('--init', default="", type=str,
                        help='"":use variance calculated by other dataset;test:use the ground truth variance for test;ce: use the celoss sensitivity')  
    args = parser.parse_args()
    savep = f"/data/advimgs/{args.victimmodel}_onlyone{args.onlyone}_dl{args.dwtlevel}_bs{args.blocksize}_max{args.budget}_eps{args.epsilon}"

    if not os.path.exists(savep) and args.saveimg==1:
        os.mkdir(savep)
        print(f"mkdir {savep}")
        
    deviceid = args.deviceid
    setSeed(args.seed)
    adbafun = ADBEvaluate(PARA_TYPE=args.paratype)
    os.environ["CUDA_VISIBLE_DEVICES"]=deviceid
    targeted = True if args.targeted == '1' else False
    early_stopping = False if args.early == '0' else True
    order = 2 if args.norm == 'l2' else np.inf
    print(args)
    torch_model =fetchImageNetModels(args.victimmodel)
    # ###############################################################################
    orig_correct_picture_num = 0
    atk_success = 0  # total number of success attack samples
    atk_success_rate = 0
    tot_queries = 0
    avg_quer = 0
    mid_quer = 0
    avg_iter = 0
    QperI = 0
    tot_iters = 0
    stop_query = []
    radius_line = [0 for i in range(args.budget + 1)]
   

    imgsize = 224
    trans = transforms.Compose([
        transforms.Resize((224,224)),
            transforms.ToTensor()
            ])
   
    cossimlist=[]
    if args.datasource =="imagenetc":

        imgbase = '../data/imagenetc'
    elif args.datasource =="imagenet":

        imgbase = '../data/imagenet'
    from PIL import Image
    celossbasepath = f"..data//resnet50_freqperturbresult"
    print(f"celossbasepath:{celossbasepath}")

    if args.init == "" or args.init=="ce":
        if args.datasource =="imagenetc":
            with open(os.path.join(imgbase,'imagenetc_test.txt'), 'r') as f:
                imagelist = f.readlines()
            imagelist = imagelist[:args.imgnum*3]
        else:
            imagelist = [f for f in os.listdir(os.path.join(imgbase,"val"))]
            imagelist.sort(key=mysortkey)
   
    ground_truth  = open(os.path.join('data/imagenet/imagenet_test.txt'), 'r').read().split('\n')
    i=0
    ttt=0
    blacklight_succ_detection,blacklight_fail_detection = 0,0
    succ_q_list = []
    all_q_list = []
    succ_l2_list = []
    all_l2_list = []
    succ_linf_list = []
    all_linf_list = []
    blacklight_detect_ratios = []
    blacklight_first_detects=[]
    blocknum_list=[]
    initquery=[]
    lowfreq_precise,freq_precise=[],[]
    if args.binaryAnalyze>0 or args.binaryAnalyze==-1:
        args.imgnum=1000
        imagelist = imagelist[2000:]
        

        ttt=2000
       
    for filename in imagelist:
        if args.datasource =="imagenet":

            if args.init=="test":
                filename = "_".join(filename.split("_")[:3])+".JPEG"
            imgpath = "{}/val/{}".format(imgbase,filename)
            ground_name_label = ground_truth[ttt]
            ttt+=1 
            
            
            ground_label =  ground_name_label.split()[1]
            ground_name =  ground_name_label.split()[0]
            innerl = 0
            while not filename==ground_name:
                ground_name_label = ground_truth[ttt]            
               
                
                ground_label =  ground_name_label.split()[1]
                ground_name =  ground_name_label.split()[0]
                innerl +=1
                if innerl>1000:
                    break
                ttt+=1
            yi = torch.tensor(int(ground_label))
            test_secrets_pil = Image.open(imgpath)
            test_secrets_pil = test_secrets_pil.convert("RGB")
            xi = trans(test_secrets_pil).unsqueeze(0)
        elif args.datasource =="imagenetc":
            ground_label = filename.strip().split(",")[1] 
            yi = torch.tensor(int(ground_label))
            imgpath = filename.strip().split(",")[0]
            test_secrets_pil = Image.open(imgpath)
            test_secrets_pil = test_secrets_pil.convert("RGB")
            xi = trans(test_secrets_pil).unsqueeze(0)

        picture_i = i
       
        original_image, label = xi, yi  # test_dataset[picture_i]
        if orig_correct_picture_num >= args.imgnum:
            break
        if i < args.beginIMG:
            i+=1
            continue
       
        if torch_model.predict_label(xi) == yi:

            if args.init =="":
                initvariables = []
             

            elif args.init=="ce":
                if not os.path.exists(os.path.join(celossbasepath,filename.split(".")[0]+"_"+str(yi.item())+"_linfceloss_list_y_randn.npy")):
                    i+=1
                    continue
                y_variance = np.load(os.path.join(celossbasepath,filename.split(".")[0]+"_"+str(yi.item())+"_linfceloss_list_y_randn.npy"))
                cb_variance = np.load(os.path.join(celossbasepath,filename.split(".")[0]+"_"+str(yi.item())+"_linfceloss_list_cb_randn.npy"))
                cr_variance = np.load(os.path.join(celossbasepath,filename.split(".")[0]+"_"+str(yi.item())+"_linfceloss_list_cr_randn.npy"))
                initvariables = [y_variance,cb_variance,cr_variance]
            else:
                initvariables = [y_variance[orig_correct_picture_num],cb_variance[orig_correct_picture_num],cr_variance[orig_correct_picture_num]]
            orig_correct_picture_num = orig_correct_picture_num + 1
         
           
            
            reres = ATK_ADBA(filename.split(".")[0],torch_model, original_image, i,
                                                                   label, picture_i, args.epsilon, 8,order, None,initvariables,savep,adbafun,args)
            if args.binaryAnalyze==10:
                continue
            elif args.binaryAnalyze==11:
                initquery.append(reres[1])
                lowfreq_precise.append(reres[-2])
                freq_precise.append(reres[-1])

                continue
            if len(reres) == 12:
                success, que, iter_num, R, R2, avgval, Rline,blacklight_count,blacklight_first_detect,chosenv_list,cossimlist,cossimavgblock_list=reres
            elif  len(reres) == 13:
                success, que, iter_num, R, R2, avgval, Rline,blacklight_count,blacklight_first_detect,chosenv_list,cossimlist,cossimavgblock_list,blocknum=reres
                blocknum_list.append(blocknum)
                continue
            if len(cossimlist)>0:
                
                if args.ablation == 0:
                    if args.binaryAnalyze==4:
                        pathp = f"../GradSignSimilarity/FGSMGrad/ourcombine_bs{args.blocksize}_{args.victimmodel}"
                    else:
                        assert args.binaryAnalyze==1
                    if not os.path.exists(pathp):
                        os.makedirs(pathp)
                    np.save(f"{pathp}/{filename.split('.')[0]}.npy",np.array(cossimlist))
                elif args.ablation == 1:
                    
                    if args.binaryAnalyze==4:
                        pathp = f"../GradSignSimilarity/FGSMGrad/ourcombine_adbasearch_bs{args.blocksize}_{args.victimmodel}"

                    if not os.path.exists(pathp):
                        os.makedirs(pathp)

                    np.save(f"{pathp}/{filename.split('.')[0]}.npy",np.array(cossimlist))


            RlineQ(Rline, radius_line, args.budget - 1)
            if success == 1 and que <= args.budget:
                atk_success = atk_success + 1
                tot_queries = tot_queries + que
                tot_iters = tot_iters + iter_num
                succ_q_list.append(que)
                succ_l2_list.append(R2)
                succ_linf_list.append(R)
                if blacklight_count>0:
                    
                    blacklight_succ_detection += 1
                    blacklight_detect_ratio = blacklight_count/que
                    blacklight_detect_ratios.append(blacklight_detect_ratio)
                    blacklight_first_detects.append(blacklight_first_detect)
                else:
                    blacklight_fail_detection += 1
            all_q_list.append(que)
            all_l2_list.append(R2)
            all_linf_list.append(R)
            atk_success_rate = atk_success / orig_correct_picture_num
            if atk_success == 0:
                avg_quer, mid_quer, avg_iter = 0.0, 0.0, 0.0
            else:
                avg_quer = tot_queries / atk_success
                stop_query.append(que)
                mid_quer = statistics.median(stop_query)
                avg_iter = tot_iters / atk_success
                QperI = avg_quer / max(avg_iter, 1)
                

            # with open("/home/code/attacks/ADBA/code/results_record/RES" + result_file_name, mode='a', encoding='utf-8', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(
            #         [picture_i, success, R, que, iter_num, atk_success_rate, avg_quer, mid_quer, avg_iter, QperI])
            print(  # f",\tIMG_{picture_i}"
                # f",\tSUCC={success}"
                # f",\tRinf={round(R, 3)}"
                # f",\tR2={round(R2,3)}"
                # f",\tQue={que}"
                # f",\tIter={iter_num}"
                f",\tACC_RATE:{round(atk_success_rate, 4)}"
                f",\tAVGquer={round(avg_quer, 3)}"
                f",\tMIDq={round(mid_quer, 1)}"
                f",\tAVGiter={round(avg_iter, 3)}"
                f",\tQperI={round(QperI, 3)}"
                f",\tSucc Avg.Q={np.mean(np.array(succ_q_list))}"
                f",\tSucc Median.Q={np.median(np.array(succ_q_list))}"
                f",\tSucc Avg.L2={np.mean(np.array(succ_l2_list))}"
                f",\tSucc Median.L2={np.median(np.array(succ_l2_list))}"
                f",\tSucc Avg.Linf={np.mean(np.array(succ_linf_list))}"
                f",\tSucc Median.Linf={np.median(np.array(succ_linf_list))}"

                f",\tAll Avg.Q={np.mean(np.array(all_q_list))}"
                f",\tAll Median.Q={np.median(np.array(all_q_list))}"
                f",\tAll Avg.L2={np.mean(np.array(all_l2_list))}"
                f",\tAll Median.L2={np.median(np.array(all_l2_list))}"
                f",\tAll Avg.Linf={np.mean(np.array(all_linf_list))}"
                f",\tAll Median.Linf={np.median(np.array(all_linf_list))}"

                
            )


        else:
            print(f"IMG{picture_i} originally classify wrongly")
        i+=1
        # if i%50==0:
        #     if len(blocknum_list)>0:
        #         if not os.path.exists(f"../GradSignSimilarity/FGSMGrad/rcolorbs{args.blocksize}_resnet50_blocknum"):
        #             os.makedirs(f"../GradSignSimilarity/FGSMGrad/rcolorbs{args.blocksize}_resnet50_blocknum")
        #         np.save(f"../GradSignSimilarity/FGSMGrad/rcolorbs{args.blocksize}_resnet50_blocknum/blocknum.npy",np.array(blocknum_list))

    # if len(blocknum_list)>0:
    #     np.save(f"../GradSignSimilarity/FGSMGrad/rcolorbs{args.blocksize}_resnet50_blocknum/blocknum.npy",np.array(blocknum_list))
    if args.binaryAnalyze==11:
        savep = f"../oursbilisearch_bs{args.blocksize}_{args.victimmodel}_initq"
        if not os.path.exists(savep):
            os.makedirs(savep)
        np.save(f"{savep}/{filename.split('.')[0]}_initquery.npy",np.array(initquery))
        np.save(f"{savep}/{filename.split('.')[0]}_lowfreq_precise.npy",np.array(lowfreq_precise))
        np.save(f"{savep}/{filename.split('.')[0]}_freq_precise.npy",np.array(freq_precise))

    print(f"ORIGINAL_CLASSIFY_ACC={orig_correct_picture_num / i}")



#BFS
def analyzeDCTSensitivity():
    target_model="vit"#
    basep=f"../data/{target_model}_freqperturbresult_0.05_5" #firstly you need to generate these images by running baselines/attack_imagenet_others.py --attack_method BFS
    chosen_channel=["y","cb","cr"]
    chosen_channel_label=["Y","Cb","Cr"]
    filelist=[f for f in os.listdir(basep)]
    linfypre_list=[[],[],[]]
    blocksize=8
    mode ="linf"#l2,linf
    for i in range(0,len(chosen_channel)):
        for filename in filelist:
            if mode=="linf":
                if filename.find(f"_linfceloss_list_{chosen_channel[i]}_randn")>-1:
                    linfpre = np.load(os.path.join(basep,filename))
                    if len(linfpre)!=blocksize**2:
                        continue
                    linfypre_list[i].append(linfpre)
            elif mode=="l2":

                if filename.find(f"_l2celoss_list_{chosen_channel[i]}_randn")>-1:
                    linfpre = np.load(os.path.join(basep,filename))
                    if len(linfpre)!=blocksize**2:
                        continue

                    linfypre_list[i].append(linfpre)

    a=np.array(linfypre_list)
    amean = np.mean(a,axis=1)
    
    plt.clf()
    for i in range(len(linfypre_list)):
        plt.plot([j for j in range(len(amean[i]))],amean[i],label=chosen_channel_label[i],linewidth=2)
    plt.xlabel("DCT channel",fontsize=24)
    plt.ylabel("CE loss",fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    plt.legend(fontsize=24)

    plt.tight_layout()
    plt.savefig(f"../data/celossDCT_randn_{target_model}_ycbcr_{mode}_bs{blocksize}.png") 




def analyzeCleanBDCT():

    trans = transforms.Compose([
        transforms.Resize((224,224)),
            transforms.ToTensor()
            ])    
    dataset= "imagenet" 
    analyzetype = "LogVariance" 
    blocksize=8
    
    savp_analyzeDiff = f"../data/{dataset}_{analyzetype}_bs{blocksize}"

    cleandir = '../data/imagenet/val'

   
    if not os.path.exists(savp_analyzeDiff):
        os.mkdir(savp_analyzeDiff)
    
    
    diffj = DiffJPEG(bs=blocksize)
    fffi = 0

    y_std_all,cb_std_all,cr_std_all =  [],[],[]
    
    filelists = [f for f in os.listdir(cleandir)]
    filelists.sort(key=mysortkey)
    ground_truth  = open(os.path.join('../data/imagenet/imagenet_test.txt'), 'r').read().split('\n')
    ttt=2000
    for filename in filelists:
        imgpath = "{}/val/{}".format(cleandir,filename)
        ground_name_label = ground_truth[ttt]
        ttt+=1 
        
        
        ground_label =  ground_name_label.split()[1]
        ground_name =  ground_name_label.split()[0]
        innerl = 0
        while not filename==ground_name:
            ground_name_label = ground_truth[ttt]            
            
            
            ground_label =  ground_name_label.split()[1]
            ground_name =  ground_name_label.split()[0]
            innerl +=1
            if innerl>1000:
                break
            ttt+=1
        yi = torch.tensor(int(ground_label))
        test_secrets_pil = Image.open(imgpath)
        test_secrets_pil = test_secrets_pil.convert("RGB")
        cleanface = trans(test_secrets_pil).unsqueeze(0)

        with torch.no_grad():
            y,cb,cr = diffj(cleanface,ycbcr=True,forged=False)        

        y_mean,y_std,y_zigzaglist = getZigzagMeanStd(y,analyzetype)
        cb_mean,cb_std,cb_zigzaglist = getZigzagMeanStd(cb,analyzetype)
        cr_mean,cr_std,cr_zigzaglist = getZigzagMeanStd(cr,analyzetype)

    
        y_std_all.append(y_std)
        cb_std_all.append(cb_std)
        cr_std_all.append(cr_std)

        fffi+=1

    data_mean = np.mean(np.array(y_std_all),axis=0)
    cbstd_mean = np.mean(np.array(cb_std_all),axis=0)
    crstd_mean = np.mean(np.array(cr_std_all),axis=0)

    data_std = np.std(np.array(y_std_all),axis=0)
    cbstd_std = np.std(np.array(cb_std_all),axis=0)
    crstd_std = np.std(np.array(cr_std_all),axis=0)


    np.save(os.path.join(savp_analyzeDiff,"y_std_all.npy"),y_std_all)
    np.save(os.path.join(savp_analyzeDiff,"cb_std_all.npy"),cb_std_all)
    np.save(os.path.join(savp_analyzeDiff,"cr_std_all.npy"),cr_std_all)
    print("finished.")



    datalist = [y_std_all,cb_std_all,cr_std_all]
    
    color_solid = ['k','#17A589','#298089']
    color_fill = ['#ABB2B9','#76D7C4','#85C1E9']
    aso=0.9
    af=0.6
    i=0
    for data in datalist:
        plt.clf()
        fig,axs = plt.subplots()


        data_mean = np.mean(data,axis=0)

        data_std = np.std(data,axis=0)


        avg_std1_large,avg_std1_small=[],[]
        for ijio in range(0,len(data_mean)):
            avg_std1_large.append(data_mean[ijio]+data_std[ijio])
            avg_std1_small.append(data_mean[ijio]-data_std[ijio])

        xais = [i for i in range(0,len(data_mean))]
        p2 = axs.plot(xais,data_mean,color =color_solid[1],alpha=aso )
        axs.fill_between(xais,avg_std1_large,avg_std1_small,alpha=af,edgecolor=color_fill[1],facecolor=color_fill[1],linewidth=0)
        p3 = axs.fill(np.NaN,np.NaN,color=color_fill[1],alpha=af)

        axs.set_ylabel("Log Variance",fontsize=24)
        axs.set_xlabel("DCT Channel",fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        plt.savefig(f"../data/channelCleanVariance{i}.jpg")
        i+=1



def pearson():
    import numpy as np
    from scipy.stats import pearsonr

    analyzetype = "LogVariance" 
    dataset="imagenet"
    target_model="vit"
    basep=f"../data/{target_model}_freqperturbresult_0.05_5" #firstly you need to generate these images by running baselines/attack_imagenet_others.py --attack_method BFS

    chosen_channel=["y","cb","cr"]
    chosen_channel_label=["Y","Cb","Cr"]
    filelist=[f for f in os.listdir(basep)]
    linfypre_list=[[],[],[]]
    blocksize=8
    mode ="linf"#l2,linf
    for i in range(0,len(chosen_channel)):
        for filename in filelist:
            if mode=="linf":
                if filename.find(f"_linfceloss_list_{chosen_channel[i]}_randn")>-1:
                    linfpre = np.load(os.path.join(basep,filename))
                    if len(linfpre)!=blocksize**2:
                        continue
                    linfypre_list[i].append(linfpre)
            elif mode=="l2":

                if filename.find(f"_l2celoss_list_{chosen_channel[i]}_randn")>-1:
                    linfpre = np.load(os.path.join(basep,filename))
                    if len(linfpre)!=blocksize**2:
                        continue
                    linfypre_list[i].append(linfpre)

    a=np.array(linfypre_list)
    amean = np.mean(a,axis=1)

    path1 = f"../data/{dataset}_{analyzetype}_bs{blocksize}"
    filename = ["y_std_all.npy","cb_std_all.npy","cr_std_all.npy"]
    
    ycbcrall,ycbcrstd=[],[]
    for name in filename:
        

        y_std_all = np.load(os.path.join(path1,name))

        ystd_mean = np.mean(np.array(y_std_all),axis=0)

        ystd_std = np.std(np.array(y_std_all),axis=0)

        ycbcrall.append(ystd_mean)
        ycbcrstd.append(ystd_std)

    r, p_value = pearsonr(amean[0], ycbcrall[0])
    cbr, cbp_value = pearsonr(amean[1], ycbcrall[1])
    crr, crp_value = pearsonr(amean[2], ycbcrall[2])

    print(f"Pearson r = {r:.4f}, p-value = {p_value:.4e}")
    print(f"cb Pearson r = {cbr:.4f}, p-value = {cbp_value:.4e}")
    print(f"cr Pearson r = {crr:.4f}, p-value = {crp_value:.4e}")




def gradSpatialPrior():
    
    spename="ResNet50_cleanimgCEloss" 
    gtgradfolder="../data/GradSignSimilarity/FGSMGrad/cleanimgCEloss_resnet50"# obtain this via baselines/attack_imagenet_others.py --attack_method fgsmgrad

    savep = f"../data/GradSignSimilarity/{spename}"
    if not os.path.exists(savep):
        os.mkdir(savep)
    blocksize=2#2,4,8,16,32
    gtgradfiles = [f for f in os.listdir(gtgradfolder)]
    gtgradfiles.sort(key=mysortkey2)
    
    blockavg_cossim,blockmedian_cossim = [],[],[],[]
    for fname in gtgradfiles:
        cossimlist,cossimlist2=[],[]
        gtgrad = torch.load(os.path.join(gtgradfolder,fname))
        gtgradsign = list(np.sign(gtgrad).flatten())
            
        p=0
        sitem = []
        gtgradsign_avgpp=[]
        
        for gi in gtgradsign:
            if (p+1)%blocksize==0:
                if p==0:
                    sitem=[gi]
                else:
                    sitem.append(gi)
                    tmpi = np.sign(np.mean(sitem))
                    gtgradsign_avgpp.append([tmpi for j in sitem])
                    cossim = torch.cosine_similarity(torch.tensor(gtgradsign_avgpp[-1]).unsqueeze(0),torch.tensor(gtgradsign[p-blocksize+1:p+1]).unsqueeze(0))
                    cossimlist2.append(float(cossim))

            else:
                if p%blocksize==0:
                    sitem=[gi]
                else:
                    sitem.append(gi)
            p+=1
        blockavg_cossim.append(np.mean(cossimlist2))
        blockmedian_cossim.append(np.median(cossimlist2))
        print(f"Others:{np.mean(cossimlist2),np.median(cossimlist2)}")            
      
    np.save(f"{savep}/blockavg_cossim_barbs{blocksize}.npy",blockavg_cossim)
    np.save(f"{savep}/blockmedian_cossim_barbs{blocksize}.npy",blockmedian_cossim)


def visgradSpatialPrior_bar():
    basep="../data/GradSignSimilarity/ResNet50_cleanimgCEloss" #obtained via gradSpatialPrior()

    plist=[
           #
            f"{basep}/blockavg_cossim_barbs4.npy",
            f"{basep}/blockavg_cossim_barbs8.npy",
            f"{basep}/blockavg_cossim_barbs16.npy",
            f"{basep}/blockavg_cossim_barbs32.npy",
            f"{basep}/blockavg_cossim_barbs2.npy"
            ]


    blockmedian_cossim_ours = [np.load(plist[i]) for i in range(0,len(plist))]


    dics,dicsloss=[],[]
    for i in range(0,len(blockmedian_cossim_ours[0])):
        dics.append({"Averaged element number":"2","Avg. Cosine Similarity":float(blockmedian_cossim_ours[4][i])})
        dics.append({"Averaged element number":"4","Avg. Cosine Similarity":float(blockmedian_cossim_ours[0][i])})
        dics.append({"Averaged element number":"8","Avg. Cosine Similarity":float(blockmedian_cossim_ours[1][i])})
        dics.append({"Averaged element number":"16","Avg. Cosine Similarity":float(blockmedian_cossim_ours[2][i])})
        dics.append({"Averaged element number":"32","Avg. Cosine Similarity":float(blockmedian_cossim_ours[3][i])})

    mypalette={"2":"#EAF1DF","4":"#FCEADC","8":"#C9D7FB","16":"#EDEAF1","32":"#DBEEF4"}

   
    plt.clf()
    df = pd.DataFrame(dics)
    sns.set(style='whitegrid',color_codes=True)
    pll = sns.violinplot(x='Averaged element number',y="Avg. Cosine Similarity",data=df,palette=mypalette)
    fig = pll.get_figure()
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel("Averaged element number",fontsize=28)
    plt.ylabel("Cosine similarity",fontsize=30)
    plt.tight_layout()
    fig.savefig(f'{basep}/violinplot.png')
    plt.close("all")




def compareInitGradCossim():
    p1="../data/FGSM_CE_origianlx"
    # firstly, you need to generate start points by using different initialization methods
    f1list=[
        "../data/GaussianInitAE", # GaussianInit() in tools/Startpoints.py
        "../data/uniforminitAE",
        "../data/targetinitAE_Resnet50",
        "../data/raysInitBinarySearchAE",
         "../data/ADBAInitAE",
         "../data/ours_dr/resnet50",
         "../data/ours_db/resnet50",
        "../data/ours_dn/resnet50"
        
    ]

    methodlist = ["Gaussian","Uniform","Image","RayS","ADBA",r"Our $\mathbf{d_r}$",r"Our $\mathbf{d_b}$",r"Our $\mathbf{d_n}$"]
    fori = '../data/imagenet/val'

    ground_truth  = open(os.path.join('../data/imagenet/imagenet_test.txt'), 'r').read().split('\n')

    trans = transforms.Compose([
        transforms.Resize((224,224)),
            transforms.ToTensor()
            ])
    dics ,dics_1= [],[]

    imglist = [f for f in os.listdir(f1list[0])]
    imglist.sort(key=mysortkey)
    ttt=2000
    celoss = torch.nn.CrossEntropyLoss()
    top_eigenvalues_list,losslist = [],[]
    i =0
    dics_1=[]
    cossimlisttotal=[]
    for fi in imglist:
        imgslist = [torch.tensor(np.load(os.path.join(f1,fi))) for f1 in f1list]

        imgori = trans(Image.open(os.path.join(fori,fi.split(".")[0]+".JPEG")).convert("RGB")).unsqueeze(0)
        
        ground_name_label = ground_truth[ttt]
        ttt+=1 
        
        
        ground_label =  ground_name_label.split()[1]
        ground_name =  ground_name_label.split()[0]
        innerl = 0
        filename = fi.split(".")[0]
        #
        while not filename==ground_name.split(".")[0]:
            ground_name_label = ground_truth[ttt]            
            
            ground_label =  ground_name_label.split()[1]
            ground_name =  ground_name_label.split()[0]
            innerl +=1
            if innerl>1000:
                break
            ttt+=1
        ground_label =torch.tensor([int(ground_label)]).cuda()

        gtpath=f"{p1}/{filename}.pth"
        if not os.path.exists(gtpath):
            print(f"File {gtpath} does not exist, skipping...")
            continue
        
        gtgrad= torch.tensor(torch.load(gtpath,weights_only=False))
        gtgradsign = torch.sign(gtgrad).flatten(start_dim=1)
        initgrad = [torch.sign(img-imgori).flatten(start_dim=1) for img in imgslist[:-1]]

        tmp = imgslist[-1].flatten(start_dim=1)
        tmp[torch.where(tmp==0)]=1
        initgrad.append(tmp)
        cossimlist = [float(torch.nn.functional.cosine_similarity(gtgradsign,initgrad[i],dim=1)) for i in range(0,len(initgrad))]
        cossimlisttotal.append(cossimlist)

        for i in range(0,len(cossimlist)):
            dics.append({"Method":methodlist[i],"Cosine similarity":float(cossimlist[i])})

    mypalette={"Gaussian":"#EAF1DF","Uniform":"#FCEADC","Image":"#C9D7FB","RayS":"#EDEAF1","ADBA":"#DBEEF4",methodlist[-3]:"#f9c59a",methodlist[-2]:"#cdc4d8",methodlist[-1]:"#E1E9FE"}
    plt.clf()
    plt.figure(figsize=(10,6))
    df = pd.DataFrame(dics)
   
    sns.set(style='whitegrid',color_codes=True)
    pll = sns.violinplot(x='Method',y="Cosine similarity",data=df,palette=mypalette)
    plt.ylabel("Cosine similarity",fontsize=30)
    plt.xlabel("Methods",fontsize=30)
    plt.xticks(fontsize=28,rotation=45)
    plt.yticks(fontsize=28,ticks=[-0.03,-0.01,0.01,0.03,0.05])
    plt.ticklabel_format(axis='y', style='sci',scilimits=(-2,-2))
    plt.gca().yaxis.get_offset_text().set_fontsize(28)
    plt.tight_layout()
    plt.savefig('../data/InitDirectionGradCosineSimilarity.png')



    print(np.mean(cossimlisttotal,axis=0))



def getdictavg(dicti):
    avgdict = {}
    for key in dicti.keys():
        avgdict[key] = np.mean(dicti[key])
    return avgdict


def fitline(x,y):
    coeff = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coeff)
    x1fit = np.linspace(min(x), max(x), 500)
    y1fit = polynomial(x1fit)
    return x1fit, y1fit

def cossimEvo():
    
    modelname = "vit"#


    p1=f"../data/GradSignSimilarity/FGSMGrad/our_pdo_{modelname}"    
    p2=f"../data/GradSignSimilarity/FGSMGrad/our_adbasearch_{modelname}"

    p3= f"../data/GradSignSimilarity/FGSMGrad/hrays_{modelname}"
    p4= f"../data/GradSignSimilarity/FGSMGrad/ADBA_ce_{modelname}"


    files = [f for f in os.listdir(p1) if f.endswith(".npy")]
    files.sort(key=mysortkey2)

    
    dict1,dict2,dict3,dict4={},{},{},{}
    for filename in files:
        x1,x2,x3,x4=[],[],[],[]
        data1 = np.load(os.path.join(p1,filename))
        data1l,data2l,data3l,data4l = [],[],[],[]
       

        for datai in data1:
           
            data1l.append(datai[0])
            x1.append(datai[1]) #query
        
        if not os.path.exists(os.path.join(p2,filename)):
            break
        data2 = np.load(os.path.join(p2,filename))
        for datai in data2:
            data2l.append(datai[0])
            
            x2.append(datai[1]) #query
        #hrays
        if not os.path.exists(os.path.join(p3,filename)):
            break
        data3 = np.load(os.path.join(p3,filename))
        for datai in data3:
            if datai[1]>=1000:
                break
            data3l.append(datai[0])
            x3.append(datai[1])

        if not os.path.exists(os.path.join(p4,filename)):
            break
        data4 = np.load(os.path.join(p4,filename))
        for datai in data4:
            if datai[1]>=1000:
                break
            data4l.append(datai[0])
            x4.append(datai[1])
        
        jj = 0
        for queryi in x1:
            if queryi not in dict1.keys():
                dict1[queryi] = [data1l[jj]]
            else:
                dict1[queryi].append(data1l[jj])
            jj += 1

        jj = 0
        for queryi in x2:
            if queryi not in dict2.keys():
                dict2[queryi] = [data2l[jj]]
            else:
                dict2[queryi].append(data2l[jj])
            jj += 1

        jj = 0
        for queryi in x3:
            if queryi not in dict3.keys():
                dict3[queryi] = [data3l[jj]]
            else:
                dict3[queryi].append(data3l[jj])
            jj += 1

        jj = 0
        for queryi in x4:
            if queryi not in dict4.keys():
                dict4[queryi] = [data4l[jj]]
            else:
                dict4[queryi].append(data4l[jj])
            jj += 1

    avgdict1 = getdictavg(dict1)
    avgdict2 = getdictavg(dict2)
    avgdict3 = getdictavg(dict3)
    avgdict4 = getdictavg(dict4)

    avgdict1_sorted = dict(sorted(avgdict1.items(), key=lambda item: item[0]))
    avgdict2_sorted = dict(sorted(avgdict2.items(), key=lambda item: item[0]))
    avgdict3_sorted = dict(sorted(avgdict3.items(), key=lambda item: item[0]))
    avgdict4_sorted = dict(sorted(avgdict4.items(), key=lambda item: item[0]))

    x1 = list(avgdict1_sorted.keys())
    x2 = list(avgdict2_sorted.keys())
    x3 = list(avgdict3_sorted.keys())
    x4 = list(avgdict4_sorted.keys())

    y1 = list(avgdict1_sorted.values())
    y2 = list(avgdict2_sorted.values())
    y3 = list(avgdict3_sorted.values())
    y4 = list(avgdict4_sorted.values())

    x1fit,y1fit = fitline(x1,y1)
    x2fit,y2fit = fitline(x2,y2)
    x3fit,y3fit = fitline(x3,y3)
    x4fit,y4fit = fitline(x4,y4)

    plt.clf()
    plt.plot(x1fit,y1fit,label="Ours",linewidth=3,color="red")
    plt.plot(x2fit,y2fit,label="Ours w/o PDO",linewidth=3)
    plt.plot(x3fit,y3fit,label="HRayS",linewidth=3)
    plt.plot(x4fit,y4fit,label="ADBA",linewidth=3)
    plt.xlabel("Query Number",fontsize=24)
    plt.ylabel("Cosine Similarity",fontsize=24)
    plt.ticklabel_format(axis='y', style='sci',scilimits=(-3,-3))
    plt.gca().yaxis.get_offset_text().set_fontsize(24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
   
    plt.tight_layout()
    plt.legend(fontsize=18)

    plt.savefig(f"../data/GradSignSimilarity/FGSMGrad/{modelname}.jpg")






def get_real_gradients(batch_size=50,dataset='CIFAR10',scale=2,modelname="resnet18"):

    if dataset == 'CIFAR10':
        from tools.fetchmodel import featchCifar10Models
        model = featchCifar10Models(modelname)          

        transform = transforms.Compose([transforms.ToTensor()])

        dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=transform)

        loader = torch.utils.data.DataLoader(dataset, batch_size=int(batch_size*scale), shuffle=True)
        

    elif dataset == 'ImageNet':
        from tools.fetchmodel import fetchImageNetModels
        model = fetchImageNetModels(modelname)
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])

        dataset = torchvision.datasets.ImageFolder(root='../data/imagenet/val', transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=int(batch_size*scale), shuffle=True)

    images, labels = next(iter(loader))
    images, labels = images.cuda(), labels.cuda()
    images.requires_grad = True


    outputs = model(images)

    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()



    pre = torch.argmax(outputs, dim=1)
    grads = images.grad.view(int(batch_size*scale), -1).cpu().numpy()


    outputs = outputs[torch.where(pre == labels)[0]][:batch_size]
    grads = grads[np.where(pre.detach().cpu().numpy() == labels.detach().cpu().numpy())[0]][:batch_size]
    images = images[torch.where(pre == labels)[0]][:batch_size]
    labels = labels[torch.where(pre == labels)[0]][:batch_size]


    return grads,images.detach()



def get_sign_consistent_blocks(grad, threshold):

    d = len(grad)
    blocks = []
    clean_signal = np.zeros(d)
    mask = np.abs(grad) >= threshold
    clean_signal[mask] = np.sign(grad)[mask]
    in_block = False

    start_idx = -1
    current_sign = 0

    for i in range(d):

        val = clean_signal[i]
        if val == 0:
            if in_block:
                blocks.append((start_idx, i))
                in_block = False

        else:

            if not in_block:

                in_block = True
                start_idx = i
                current_sign = val

            else:
                if val != current_sign:

                    blocks.append((start_idx, i))
                    start_idx = i
                    current_sign = val

    if in_block:
        blocks.append((start_idx, d))

    return blocks




def count_sign_runs(vec):
    """
    vec: 1D numpy array with values in {-1, +1}
    return: number of sign-consistent runs
    """
    if len(vec) == 0:
        return 0
    return 1 + np.sum(vec[1:] != vec[:-1])


@torch.no_grad()

def analyze_costs(grads, images, sparsity_levels=[0.02, 0.05, 0.10]):


    d = grads.shape[1]
    results = {}

    if images.shape[2] == 224:
        blocksize = 4
    elif images.shape[2] == 32:
        blocksize = 2

    diffj = DiffJPEG(bs=blocksize) 

    for sp in sparsity_levels:
        print(f"Testing Sparsity: Top {sp*100}%")
        dyadic_list = []
        pattern_list = []

        all_block_lengths_pool = [] 

        imgi = 0
        for i, g in enumerate(grads):
            original_image_x = images[imgi].unsqueeze(0)

            threshold = np.percentile(np.abs(g), 100 * (1 - sp))

            true_blocks = get_sign_consistent_blocks(g, threshold) 

            if len(true_blocks) == 0: continue


            yc,cb,cr = diffj(original_image_x.detach().cpu(),forged=False,batch=True)

            ycbcr = torch.cat([yc.unsqueeze(1),cb.unsqueeze(1),cr.unsqueeze(1)],dim=1)

            n_block = ycbcr.shape[2]

            y_mu,y_std,_ = getZigzagMeanStd(yc[0])
            cb_mu,cb_std,_ = getZigzagMeanStd(cb[0])
            cr_mu,cr_std,_ = getZigzagMeanStd(cr[0])

            stds =[y_std,cb_std,cr_std]

        
            newd,newcandi,newdis,newdcolor = getnewd(ycbcr,n_block,stds,original_image_x,npop=1,nchannel=original_image_x.shape[1],\
                                        step_p=1,diffj=diffj,ord=np.inf,blocksize=blocksize) 

            zeronum = len(torch.where(torch.sign(newd)==0)[0])
            print(f"newd, zeronum:{zeronum}")

            if zeronum>0:
                newd[torch.where(newd==0)]=1
                d0 = torch.sign(newd)
                d0[torch.where(d0==0)] = 1
                d0 = d0.reshape(1, -1).cpu().numpy()[0]
            else:
                d0 = torch.sign(newd).reshape(1, -1).cpu().numpy()[0]

            
            current_img_dyadic_costs = []
            for start, end in true_blocks:
                length = end - start
                all_block_lengths_pool.append(length) 

                #  log2(d/length)
                cost = np.log2(d / max(length, 1)) 
                current_img_dyadic_costs.append(cost)


            if len(current_img_dyadic_costs) > 0:
                dyadic_list.append(np.mean(current_img_dyadic_costs))

            # 2.  Pattern Cost (Gamma)
            current_img_gamma_counts = []
            d0_np = d0  # already numpy, shape [d]

            for (start, end) in true_blocks:
                segment = d0_np[start:end]

                if len(segment) == 0:
                    continue

                gamma_k = count_sign_runs(segment)
                current_img_gamma_counts.append(gamma_k)
                

            if len(current_img_gamma_counts) > 0:
                pattern_list.append(np.mean(current_img_gamma_counts))

            imgi += 1
            print(f"Processed {imgi}/{len(grads)} pictures: Pattern Cost = {np.mean(current_img_gamma_counts):.4f}, Dyadic Cost = {np.mean(current_img_dyadic_costs):.4f}")

            


        print(
            f"|B| avg={np.mean(all_block_lengths_pool):.2f}, "
            f"gamma avg={np.mean(current_img_gamma_counts):.2f}"
        )

        results[sp] = {
            'dyadic_mean': np.mean(dyadic_list),
            'pattern_mean': np.mean(pattern_list),
            'dyadic_std': np.std(dyadic_list),
            'pattern_std': np.std(pattern_list),
            'len_mean': np.mean(all_block_lengths_pool),
            'len_std': np.std(all_block_lengths_pool)

        }

        print(f"Top {sp*100}%: Avg Block Len={results[sp]['len_mean']:.2f}, Dyadic={results[sp]['dyadic_mean']:.2f}")


    return results



def callTheorem5():


    modelname = 'vit' # resnet18,wrn,vgg;regnet,SwinV2T,ConvNextBase,inv3,vit,efficient,deit,dense121,wrs50#resnet18,vgg,wrn
    dataset = 'ImageNet' # CIFAR10  'ImageNet'

    batchsize = 100
    scale = 1.5


    real_grads, images = get_real_gradients(batch_size=batchsize, dataset=dataset, scale=scale,modelname=modelname)
    sparsity_levels = [0.02, 0.05, 0.10, 0.20, 0.50, 0.70, 0.90, 1.0]
    filep_json =  f'../data/theorem5figs/runs_distribution_{dataset}_{modelname}_real_gradients_dualaxis_correctgamma.json'

    if not os.path.exists(filep_json):
        data = analyze_costs(real_grads, images, sparsity_levels=sparsity_levels)
        with open(filep_json, 'w') as json_file:

            def convert(o):
                if isinstance(o, np.int64): return int(o)
                if isinstance(o, np.float32): return float(o)
                if isinstance(o, np.float64): return float(o)  
                raise TypeError
            json.dump(data, json_file, indent=4, default=convert)
    else:
        with open(filep_json, 'r') as json_file:
            data =json.load(json_file)


    x_labels = [f"{float(k)*100:g}" for k in data.keys()]
    x_pos = np.arange(len(x_labels)) 

    y_dyadic = [v['dyadic_mean'] for v in data.values()]
    err_dyadic = [v['dyadic_std'] for v in data.values()]

    y_pattern = [v['pattern_mean'] for v in data.values()]
    err_pattern = [v['pattern_std'] for v in data.values()]

    y_len = [v['len_mean'] for v in data.values()]
    err_len = [v['len_std'] for v in data.values()]

    plt.clf()
    fig, ax1 = plt.subplots(figsize=(9, 6))


    ax1.set_xlabel('Top Magnitude Gradient %', fontsize=30)
    ax1.set_ylabel('Average Complexity', fontsize=30)

    lns1 = ax1.errorbar(x_pos, y_dyadic, yerr=err_dyadic, fmt='-o', capsize=8, 
                 label=r'Dyadic Cost ($\log_2(d/|B|)$)', color='tab:red', linewidth=5)

    lns2 = ax1.errorbar(x_pos, y_pattern, yerr=err_pattern, fmt='-s', capsize=8, 
                 label=r'Pattern Cost ($\gamma$)', color='tab:blue', linewidth=5)

    ax1.tick_params(axis='y', labelsize=30)
    ax1.tick_params(axis='x', labelsize=30)

    ax2 = ax1.twinx() 
    ax2.set_ylabel('Block Length $|B|$', fontsize=30, color='tab:green')
    lns3 = ax2.errorbar(x_pos, y_len, yerr=err_len, fmt='--^', capsize=8, \
                 label=r'Block Length ($|B|$)', color='tab:green', linewidth=5)

    ax2.tick_params(axis='y', labelcolor='tab:green', labelsize=30)
    lns = [lns1, lns2, lns3]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', fontsize=24, framealpha=0.5,bbox_to_anchor=(0,0.9))
    plt.xticks(x_pos, x_labels) 
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    filep_png =f'../data/theorem5figs/runs_distribution_{dataset}_{modelname}_real_gradients_dualaxis_correctgamma_larger.png'

    plt.savefig(filep_png, dpi=300)

    print(f"save pictures to {filep_png}")



if __name__ == "__main__":
    main_ADBA()
