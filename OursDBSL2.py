# coding:utf-8
import os 
import math
import argparse,math
import torch
import numpy as np
import copy,pywt
#import csv
import random
#from PIL import Image
from torchvision import transforms
import sys
#from datetime import datetime
from torchvision.utils import save_image
from tools.jpegdct import DiffJPEG
from tools.fetchmodel import fetchImageNetModels

from tools.DataTools import ADBEvaluate

import statistics

from models.OursClass import Block,V,getzigzagcor,getZigzagMeanStd,getNewdRays
from tools.utils import setSeed,progress_bar
##################################################################################################
#represent blocks of a picture
class Iter:
    def __init__(self, init_vbest, offspringN, iter_n=1,early_stop=False,tracker=None,paratype=10,useadba=1,l2update=0):
        self.offspringN = offspringN
        self.iter_n = iter_n
        self.l2update = l2update

        self.offspringVs = [] #d1 d2
        self.early_stop = early_stop
        self.chosen_v = -1 #
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
    def mutation(self, bestl2,model, original_image, label, aim_r, tolerance_binary_iters, blocks, binaryM,method,globalq=0,hisv=[],hisblock=[],hispre=[],gtgrad=None,nonzeroidx=None):
        query = 0
        self.dim = original_image.shape[1] * original_image.shape[2] * original_image.shape[3]
        cossim=[]
        for vi in range(self.offspringN):
            self.offspringVs[vi].reverse_v(blocks[vi])
            self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = self.old_vbest.Rmax, 0.0

        orioffspringlen = len(self.offspringVs)
       
        query_plus,hisv,hisblock,hispre,cossim =  self.compare_directions_usingADB(bestl2,
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



    def compare_directions_usingADB(self, bestl2,model, original_image, label, aim_r, maxIters, binaryM,method,globalq=0,hisv=[],hisblock=[],blocks=[],hispre=[],orioffspringlen=2,gtgrad=None,cossim=[]):
        perturbations = [] #d1 d2
        perturbed_images = [] #= x+ADB*d
        predicted = [] #= F(x+ADB*d)
        query = 0
        succV = []
        self.chosen_v = -1
        
        
        for i in range(len(self.offspringVs)):
            perturbations.append(self.offspringVs[i].advv_to_tensor().cuda())
            tmpa = projl2(original_image,perturbations[i],self.old_vbest.Rmax) 
            diff = tmpa-original_image
            perturbed_images.append(tmpa)

            
            predicted.append(model.predict_label(perturbed_images[i]).cpu())

            if self.tracker is not None:
                match_num = self.tracker.add_img(perturbed_images[i][0].detach().cpu().numpy())
                if match_num>self.blacklight_threshold:
                    self.blacklight_count+=1
                    if self.blacklight_first_detect ==0:
                        self.blacklight_first_detect = globalq+query + 1
            query = query + 1
            if (predicted[i] != label and (torch.norm(diff)<bestl2 and self.l2update==1)) or (self.l2update==0 and predicted[i] != label):
                succV.append(i)
                self.offspringVs[i].Rmax = self.old_vbest.Rmax
                self.chosen_v = i
                # if self.early_stop:
                #     break
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
                tmpa = torch.clamp(
                    projl2(original_image,perturbations[succVtemp[vi]],ADB), 0.0, 1.0) 
                diff = tmpa-original_image


                perturbed_images[succVtemp[vi]] =tmpa
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
                    dis = torch.norm(perturbed_images[succVtemp[vi]]-original_image, p=2)

                    if dis <= aim_r:
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
            #if ite >= 10 :
                break

        #d1 and d2 are close, just return d1
        self.chosen_v = succV[0] 
        if gtgrad is not None:
            cossim=[float(torch.cosine_similarity(torch.tensor(self.offspringVs[self.chosen_v].adv_v).unsqueeze(0),torch.sign(gtgrad).cpu().flatten(start_dim=1)))]

        return query,hisv,hisblock,hispre,cossim



def getNewd(ycbcr,n_block,stds,image,npop,nchannel,step_p,diffj,ord,initmu=0,\
            initystd=0,initcbstd=0,initcrstd=0,blocksize=8,init="",initvariables=[],\
                freqratio=1,color=0,returnColorimg=False):

    y_std,cb_std,cr_std = stds[0],stds[1],stds[2]
    n_block = int((224/blocksize)**2)
    mu = torch.zeros(1,3,n_block,blocksize,blocksize)
    mu[:,:,:,0,0] += initmu
    modifys = torch.zeros(npop,nchannel,n_block,blocksize,blocksize)
    mu_zs = torch.zeros(npop,nchannel,n_block,blocksize,blocksize)
    cosa =torch.tensor([10000.]).cuda()
    samplei = 0
    repeat_num = int(n_block//n_block)
   
    color=0
    stepp=1 
    
    step_p_y = stepp #[block_size:1:1e4;2: 100;4:1;8:0.01]
    #for color
    step_p_cb_color = stepp*100 
    step_p_cr_color = stepp*100
    step_p_cb = 0
    step_p_cr = 0
    perturb_range = blocksize**2
    
    print(f"perturb_channel end at:{perturb_range},step_p_y:{step_p_y},step_p_cb:{step_p_cb},step_p_cr:{step_p_cr}")
    for i in range(perturb_range):
    #for i in range(14,15):
        zigzagi,zigzagj = getzigzagcor(i,rows=blocksize,columns=blocksize)
        mu_z = torch.randn((npop,nchannel,n_block))
        if not init=="":
            stdbias = torch.cat([(mu_z[:,0,:]*(y_variance[i])).unsqueeze(1),\
                                            (mu_z[:,1,:]*(cb_variance[i])).unsqueeze(1),\
                                            (mu_z[:,2,:]*(cr_variance[i])).unsqueeze(1)],dim=1)
        else:   
            stdbias = torch.cat([(mu_z[:,0,:]*(y_std[i]**2)).unsqueeze(1),\
                                (mu_z[:,1,:]*(cb_std[i]**2)).unsqueeze(1),\
                                (mu_z[:,2,:]*(cr_std[i]**2)).unsqueeze(1)],dim=1)
        
        modify = mu.repeat(npop,1,1,1,1)[:,:,:,zigzagi,zigzagj]+stdbias
        
        modifys[:,:,:,zigzagi,zigzagj] = modify

    modifys[:,0,:,:,:] *= step_p_y

    modifys_color = modifys.clone()

    modifys[:,1,:,:,:] *= step_p_cb
    modifys[:,2,:,:,:] *= step_p_cr

    modifys_color[:,1,:,:,:] *= step_p_cb_color
    modifys_color[:,2,:,:,:] *= step_p_cr_color
    
    ycbcr_pert = ycbcr.clone()+ modifys.repeat(1,1,repeat_num,1,1)
    
    adv_images = diffj.rec(ycbcr_pert[:,0],ycbcr_pert[:,1],ycbcr_pert[:,2],image.shape[2],image.shape[3]) 

    adv_images = adv_images.cuda()
    
    adv_images = torch.clamp(adv_images,0,1)  

    perturb_pixel =  adv_images-image 
    dis = torch.norm(perturb_pixel,p=ord)
    newd = perturb_pixel/torch.norm(perturb_pixel)


    ycbcr_pert_color = ycbcr.clone()+ modifys_color.repeat(1,1,repeat_num,1,1)        
    adv_images_color = diffj.rec(ycbcr_pert_color[:,0],ycbcr_pert_color[:,1],ycbcr_pert_color[:,2],image.shape[2],image.shape[3]) 
    adv_images_color = adv_images_color.cuda()
    adv_images_color = torch.clamp(adv_images_color,0,1)  
    perturb_pixel_color =  adv_images_color-image 
    dis_color = torch.norm(perturb_pixel_color,p=ord)
    newd_color = perturb_pixel_color/torch.norm(perturb_pixel_color)

    
    if returnColorimg:
        return newd,adv_images,dis,newd_color,adv_images_color
    else:
        return newd,adv_images,dis,newd_color


def projl2(x,x2,alphas):

    alphas = alphas*torch.tensor(np.ones(x.shape)).float()
    return torch.clamp((1-alphas) * x+ alphas * x2.cpu(),0,1)

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
    newd_image_binary_list,v0_list = [],[]
    blocksize_list = [2,4,8,16]
    blocksize_list_ori = blocksize_list
    chosenv_list = []
    newcandi_list=[]
    for blocksize in blocksize_list:
        diffj = DiffJPEG(bs=blocksize)
        yc,cb,cr = diffj(original_image_x.detach().cpu(),forged=False,batch=True)
        ycbcr = torch.cat([yc.unsqueeze(1),cb.unsqueeze(1),cr.unsqueeze(1)],dim=1)
        n_block = ycbcr.shape[2]
        y_mu,y_std,_ = getZigzagMeanStd(yc[0])
        #print(list(y_std[0:8]))
        cb_mu,cb_std,_ = getZigzagMeanStd(cb[0])
        cr_mu,cr_std,_ = getZigzagMeanStd(cr[0])
        stds =[y_std,cb_std,cr_std]
        mus = [y_mu,cb_mu,cr_mu]
        newd,newcandi,newdis,newd_color,newcandi_color = getNewd(ycbcr,n_block,stds,original_image_x.cuda(),npop=npop,nchannel=nchannel,\
                                    step_p=step_p,diffj=diffj,ord=order,initmu=args.mu,initystd=args.initystd,\
                                        initcbstd=args.initcbstd,initcrstd=args.initcrstd,blocksize=blocksize,\
                                            init=args.init,initvariables=initvariables,freqratio=args.freqratio,\
                                                color = args.color,returnColorimg=True) 
        newcandi_list.append(newcandi)


    query = 0
    initrhigh=1
    initrlow=0
    success = -1
    candi = None
    
    thislimit = min(10,args.budget) 
    thislooplimit = 10
    print(f"thislooplimit:{thislooplimit}")
    thisloop = 0
    rangescale = initrhigh-initrlow
    disbest = 100000
    if args.budget<=20:
        args.earlyexit = 0
    print(f"earlyexit:{args.earlyexit}")

    while thisloop<thislooplimit and query< args.budget:
        if len(newcandi_list)==1 and args.earlyexit==1:
            if args.onlyone==0:
                break
        mid = (initrhigh+initrlow)/2
        rangescale = initrhigh-initrlow
        succlist,dis_list=[],[]
        for newcandi in newcandi_list:
            candi = torch.clamp(projl2(original_image_x,newcandi,mid),0,1)
            pre = torch.argmax(model(candi.cuda())).cpu()
            query+=1
            dis = torch.norm(candi-original_image_x,p=2)
            succlist.append((pre!=label_y).item())
            dis_list.append(float(dis))
            if dis<= aim_r and pre!=label_y:
                success = 1
                Rbest=dis
                print(f"succ blocksize:{blocksize_list}")
                globalblocksize.extend(blocksize_list)

                adv_img = candi[0].unsqueeze(0).cuda()-original_image_x.cuda()
                Rline = [[0, 1.0]]
                nparray = np.array(adv_img.cpu()).flatten()
                return success, query, 0, Rbest, np.linalg.norm(nparray, ord=2), np.mean(
                    nparray), Rline ,0,0,[],[],[] # np.linalg.norm(nparray,ord=np.inf)
                
        failidx = np.where(np.array(succlist)==False)[0]
        if len(failidx)==len(newcandi_list): 
            initrlow = mid 
        else:
            tmp = [i for num,i in enumerate(newcandi_list) if num not in failidx]
            if len(tmp)==0 and len(newcandi_list)==1:
                break

            newcandi_list = tmp 
            tmp = [i for num,i in enumerate(blocksize_list) if num not in failidx]
            blocksize_list = tmp
            tmp = [i for num,i in enumerate(v0_list) if num not in failidx]
            v0_list = tmp

            
            initrhigh = mid 
            
        thisloop+=1
    if thisloop>=thislooplimit:
        print("warmup loop limit reached")
        print(f"range scale;{rangescale},initrhigh:{initrhigh},initrlow:{initrlow}")

    blocksize = np.random.choice(blocksize_list)
    print(f"blocksize_list:{blocksize_list}")
    choseidx = blocksize_list_ori.index(blocksize)
    usz = np.where(np.array(blocksize_list)==blocksize)[0]
    newd_image_binary = newcandi_list[usz[0]].cpu()
    tmpd = newd_image_binary-original_image_x

    candi = torch.clamp(projl2(original_image_x,newd_image_binary,initrhigh),0,1)
    disbest = torch.norm(candi-original_image_x,2)

    zeronum = len(torch.where(torch.sign(tmpd)==0)[0])
    print(f"final newd, zeronum:{zeronum}")
    if zeronum>0:
        tmpd[torch.where(tmpd==0)]=1
        sign_new_low = torch.sign(tmpd)
        sign_new_low[torch.where(sign_new_low==0)] = 1
        adv_v_init = list(np.array(sign_new_low[0].flatten().detach().cpu().numpy(),dtype=np.int32))
    else:
        adv_v_init = list(np.array(torch.sign(tmpd[0]).flatten().detach().cpu().numpy(),dtype=np.int32))


    v0 = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=adv_v_init )

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


        if args.onlyone!=1:
            if dwtlevel>0:
                hshape,wshape = cas.shape[2],cas.shape[3]
            else:
                hshape,wshape =original_image_x.shape[2],original_image_x.shape[3]
                dwtblocksize = args.blocksize
            
            if args.lowtype=="rcolor":
                randomint_tmp = np.random.randint(0,256,(npop,3,hshape//dwtblocksize,wshape//dwtblocksize) )
                    
                randomint = torch.tensor(randomint_tmp).unsqueeze(4).unsqueeze(5).repeat(1,1,1,1,dwtblocksize,dwtblocksize)
                randomint = randomint.permute(0,1,2,4,3,5).flatten(start_dim=-2)
                randomint = randomint.permute(0,1,4,2,3).flatten(start_dim=-2)
                randomint = randomint/255


                todonp = randomint.detach().cpu().numpy() 

            elif args.lowtype=="dwtstd":
                todonp = torch.nn.functional.interpolate(newcandi_color,(hshape,wshape)).cpu()

            if dwtlevel>0 and args.lowtype!="dct":
                cas = torch.tensor(cas)
                global casmin,casmax,casNormalize
                casmin= float(torch.min(cas))
                casmax = float(torch.max(cas))
                casNormalize = (cas-casmin)/(casmax-casmin)

                cHs_repeat = torch.tensor(cHs).repeat(npop,1,1,1).numpy()
                cVs_repeat = torch.tensor(cVs).repeat(npop,1,1,1).numpy()
                cDs_repeat = torch.tensor(cDs).repeat(npop,1,1,1).numpy()
                #todonp_unnorm = torch.clamp(casNormalize*0.5 + todonp*0.5,0,1)
                if  args.replace==0:
                    todonp_unnorm = torch.clamp(casNormalize*0.5 + todonp*0.5,0,1)
                    todonp_unnorm = todonp_unnorm*(casmax-casmin)+casmin
                else:
                    todonp_unnorm = todonp*(casmax-casmin)+casmin


                if dwtlevel==1:
                
                    #recons = pywt.idwt2((todonp_unnorm,(cHs_repeat,cVs_repeat,cDs_repeat)),'haar')
                    recons = pywt.idwt2((todonp_unnorm,(np.zeros_like(cHs_repeat),np.zeros_like(cVs_repeat),np.zeros_like(cDs_repeat))),'haar')

                elif dwtlevel==2:
                    cHs_0_repeat = torch.tensor(cHs_0).repeat(npop,1,1,1).numpy()
                    cVs_0_repeat = torch.tensor(cVs_0).repeat(npop,1,1,1).numpy()
                    cDs_0_repeat = torch.tensor(cDs_0).repeat(npop,1,1,1).numpy()

                    # recons_1 = pywt.idwt2((todonp_unnorm,(cHs_repeat,cVs_repeat,cDs_repeat)),'haar')
                    # recons = pywt.idwt2((recons_1,(cHs_0_repeat,cVs_0_repeat,cDs_0_repeat)),'haar')
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
                dis_low = torch.norm(perturb_pixel,p=2)
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

    if args.onlyone!=1:
        newd_image_low = []
        for v0_low in v0_low_list:
            newd_image_low_tmp = v0_low.advv_to_tensor()
            newd_image_low.append(newd_image_low_tmp)
        newd_image_low = torch.stack(newd_image_low)
    bestl2=disbest

    if args.onlyone==0:
        
        with torch.no_grad():
            candi_low = torch.clamp(projl2(original_image_x,recons,initrhigh),0,1)
            pre_low = torch.argmax(model(candi_low.cuda())).cpu()
            query += 1
            dis_low = torch.norm(candi_low-original_image_x,p=2)
            if pre_low==label_y :

                candi = torch.clamp(projl2(original_image_x,newd_image_binary,initrhigh),0,1)
                args.paratype=22
            else:

                globalquery = query 
                query = 0
                initrhigh_low=initrhigh
                initrlow,initrlow_low=0,0
                success = -1
                candi = None
                chosenv_list = []
                flag_stop ,flag_low_stop = 0,0
                query_1,query_1_low = 0,0
                innerqlimit = tolerance_binary_iters
                print(f"innerqlimit:{innerqlimit}")
                innerloop = 0
                stop1,stop2 = False,False
                prev_dis_low,prev_dis =dis_low,disbest
                if dis_low<disbest:
                    disbest = dis_low

                while query<innerqlimit*2 and query <args.budget and success==-1 and abs(query_1-query_1_low)<2 and not stop1 and not stop2:

                    if  query_1_low<innerqlimit and (dis_low == disbest or flag_low_stop<2):
                        stop2 = False
                        mid_low = initrhigh_low-(initrhigh_low-initrlow_low)/5
                        if args.lowtype=="dct":
                            candi_low = torch.clamp(lowpassimg+mid_low*newd_image_color,0,1)
                        elif args.lowtype=="rcolor" or args.lowtype=="bar" or args.lowtype=="dwtstd":
                            candi_low = torch.clamp(projl2(original_image_x,recons,mid_low),0,1)

                        pre_low = torch.argmax(model(candi_low.cuda())).cpu()
                        query_1_low += 1
                        query += 1
                        dis_low = torch.norm(candi_low-original_image_x,p=2)
                        if pre_low==label_y:
                            flag_low_stop +=1
                            initrlow_low = mid_low
                            tmp = prev_dis_low
                        else:
                            initrhigh_low = mid_low
                            #disbest = dis_low 
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

                    if query_1<innerqlimit and (dis == disbest or  flag_stop<2) :
                        stop1=False
                        mid = initrhigh-(initrhigh-initrlow)/5
                        candi = torch.clamp(projl2(original_image_x,newd_image_binary,mid),0,1)

                        pre = torch.argmax(model(candi.cuda())).cpu()
                        query_1 += 1
                        query += 1
                        dis = torch.norm(candi-original_image_x,p=2)
                        if pre==label_y:
                            flag_stop +=1
                            initrlow = mid
                            tmp = prev_dis
                        else:
                            initrhigh = mid 
                            #disbest = dis
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
                    # if flag_low_stop and flag_stop:
                    #     break
                    innerloop += 1
                    if innerloop >= 4 and (initrhigh_low - initrlow_low <= 0.0002 or initrhigh - initrlow <= 0.0002):
                    #if ite >= 10 :
                        print("loop threshold.")
                        break
                tmp = torch.clamp(projl2(original_image_x,newd_image_binary,initrhigh),0,1)

                if args.lowtype=="dct":
                    tmp_low = torch.clamp(lowpassimg+initrhigh_low*newd_image_color,0,1)
                elif args.lowtype=="rcolor" or args.lowtype=="bar"  or args.lowtype=="dwtstd":
                    tmp_low = torch.clamp(projl2(original_image_x,recons,initrhigh_low),0,1)

                dis = torch.norm(tmp-original_image_x,p=2)
                dis_low = torch.norm(tmp_low-original_image_x,p=2)
                print(f"dis:{float(dis)},dis_low:{float(dis_low)}")


                if dis_low<=dis: 
                    candi = tmp_low
                    initrhigh = initrhigh_low
                    bestl2 = dis_low
                    print(f"update bestl2 as dis_low:{dis_low}")

                else:
                    candi = tmp
                    bestl2 = dis
                    print(f"update bestl2 as dis:{dis}")

                tmpd = candi-original_image_x
                zeronum = len(torch.where(torch.sign(tmpd)==0)[0])
                print(f"final newd, zeronum:{zeronum}")
                if zeronum>0:
                    tmpd[torch.where(tmpd==0)]=1
                    sign_new_low = torch.sign(tmpd)
                    sign_new_low[torch.where(sign_new_low==0)] = 1
                    adv_v_init = list(np.array(sign_new_low[0].flatten().detach().cpu().numpy(),dtype=np.int32))
                else:
                    adv_v_init = list(np.array(torch.sign(tmpd[0]).flatten().detach().cpu().numpy(),dtype=np.int32))


                v0 = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=adv_v_init )
                args.paratype=22

                globalquery+=query 
                query = globalquery
            if gtgrad is not None:
                cossimlist_init = float(torch.cosine_similarity(torch.sign(candi.cpu()-original_image_x.cpu()).flatten(start_dim=1).cpu(),torch.sign(gtgrad).flatten(start_dim=1).cpu(),dim=1))#等于1
            else:
                cossimlist_init = None

    print(f"args.paratype:{args.paratype}")
    
    if cossimlist_init is not None:
        cossimlist = [[cossimlist_init,query]]
        cossimavgblock_list=[]
    else:
        cossimlist,cossimavgblock_list = [],[]

    with torch.no_grad():
        if args.saveimg==1:
            save_image(tmp,f'{savep}/{img_number}_{int(label_y)}_{query}.png')            
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

            
            Rline = [[0, bestl2]]
            v0.Rmax =initrhigh
            ITERATION = Iter(v0, args.offspringN, 1,early_stop=args.early_stop,tracker=tracker,paratype=args.paratype,useadba=args.useadba,l2update=args.l2update)
            progress_bar(img_number, query, iter_num, bestl2)
            """"""
            singleNext = False
            hisv,hisblock,hispre=[],[],[] 
            blockdir1_stop = False
            while (query < args.budget) and (bestl2 > aim_r):  

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
                    
                    query_plus,hisv,hisblock,hispre,cossim,cossimavgblock = ITERATION.mutation(bestl2,model, original_image_x, label_y, aim_r, tolerance_binary_iters,
                                                    blocks_i[args.offspringN * i:args.offspringN * (i + 1)], args.binaryM,method=method,globalq=query,\
                                                        hisv=hisv,hisblock=hisblock,hispre=hispre,gtgrad=gtgrad,nonzeroidx=nonzeroidx)
                    if len(cossim)>0:
                        cossimlist.append([cossim[0],query])
                    if len(cossimavgblock)>0:
                        cossimavgblock_list.append(cossimavgblock)
                    tmp_Rbest = ITERATION.old_vbest.Rmax
                    tmp_adversarial_v = ITERATION.old_vbest.advv_to_tensor()
                    tmp_adversarial_image = projl2(original_image_x,tmp_adversarial_v,tmp_Rbest)
                    bestl2 = torch.norm(tmp_adversarial_image-original_image_x,p=2)                  

                    query = query + query_plus
                    Rline.append([query, bestl2])
                    if args.saveimg==1:
                        tmp_Rbest = ITERATION.old_vbest.Rmax
                        tmp_adversarial_v = ITERATION.old_vbest.advv_to_tensor()
                        tmp_adversarial_image = original_image_x + tmp_Rbest * tmp_adversarial_v
                        tmp_adversarial_image = torch.clamp(tmp_adversarial_image, 0.0, 1.0)                   
                        save_image(tmp_adversarial_image,f'{savep}/{img_number}_{int(label_y)}_{query}.png')

                    iter_num = iter_num + 1
                    if (bestl2 <= aim_r) or query >= args.budget:
                        break
                if len(blocks_i)>0:
                    blocks.append(copy.deepcopy(blocks_i))


            if  (query < args.budget) and (ITERATION.old_vbest.Rmax > aim_r):
                innerloop=0
                diffj = DiffJPEG(bs=blocksize)
                yc,cb,cr = diffj(original_image_x.detach().cpu(),forged=False,batch=True)
                ycbcr = torch.cat([yc.unsqueeze(1),cb.unsqueeze(1),cr.unsqueeze(1)],dim=1)
                n_block = ycbcr.shape[2]
                y_mu,y_std,_ = getZigzagMeanStd(yc[0])
                #print(list(y_std[0:8]))
                cb_mu,cb_std,_ = getZigzagMeanStd(cb[0])
                cr_mu,cr_std,_ = getZigzagMeanStd(cr[0])
                stds =[y_std,cb_std,cr_std]
                mus = [y_mu,cb_mu,cr_mu]
                newd_rev,newcandi,newdis,newd_color = getNewd(ycbcr,n_block,stds,original_image_x.cuda(),npop=npop,nchannel=nchannel,\
                                            step_p=step_p,diffj=diffj,ord=order,initmu=args.mu,initystd=args.initystd,\
                                                initcbstd=args.initcbstd,initcrstd=args.initcrstd,blocksize=blocksize,\
                                                    init=args.init,initvariables=initvariables,freqratio=args.freqratio,\
                                                        color = args.color) 
                
                
                
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
                        tmp_Rbest = ITERATION.old_vbest.Rmax
                        tmp_adversarial_v = ITERATION.old_vbest.advv_to_tensor()
                        tmp_adversarial_image = projl2(original_image_x,tmp_adversarial_v,tmp_Rbest)
                        bestl2 = torch.norm(tmp_adversarial_image-original_image_x,p=2)  
  
                        query = query + query_plus
                        Rline.append([query, bestl2])
                        iter_num = iter_num + 1
                        if (bestl2 <= aim_r) or query >= args.budget:
                            break
                    if len(blocks_i)>0:
                        blocks.append(copy.deepcopy(blocks_i))
                  
                innerloop += 1
            Rbest = ITERATION.old_vbest.Rmax
            adversarial_v = ITERATION.old_vbest.advv_to_tensor()
            adversarial_image = projl2(original_image_x, adversarial_v, Rbest)
            adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)
            

            success = 1
            if Rbest > aim_r:
                success = -1
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
    parser.add_argument('--victimmodel', default='dense121', type=str,
                        help='ConvNextBase,SwinV2T,dense121,efficient,vit,inv3,wrs50,resnet50')#
    parser.add_argument('--apitype', default='standard', type=str,
                        help='')#

    parser.add_argument('--replace', default=0, type=int,
                        help='Dataset')
    parser.add_argument('--earlyexit', default=0, type=int,
                        help='Dataset')

    parser.add_argument('--optimal', default=0, type=int,
                        help='for rebuttal,run the result with optimal para')
    parser.add_argument('--warmup', default=0, type=int,
                        help='0: dynamicly choose blocksize for every image; \
                            1: warmup to choose the most frequent blocksize of the initial several images;\
                                -1: have warmed up')
    parser.add_argument('--warmupsize', default=20, type=int,
                        help='')
    parser.add_argument('--l2update', default=0, type=int,
                        help='0::None,1:adbasearch.')


    parser.add_argument('--useadba', default=0, type=int,
                        help='Dataset')
    parser.add_argument('--binaryAnalyze', default=0, type=int,
                        help='Dataset')
    parser.add_argument('--onlyone', default=0, type=int,
                        help='0::both,1:only std sample;2:only low color square.')
    parser.add_argument('--ablation', default=0, type=int,
                        help='0::None,1:adbasearch.')
    parser.add_argument('--lowtype', default="rcolor", type=str,
                        help='dct,bar,rcolor,dwtstd') 
    parser.add_argument('--dwtlevel', default=4, type=float,
                        help='0:no;1;2')  
    parser.add_argument('--dctTrunc', default=1, type=float,
                        help='1 to blocksize**2')  


    parser.add_argument('--epsilon', default=5, type=float,
                        help='attack strength')
    parser.add_argument('--gradnpop', default=10, type=int,
                        help='attack strength')
    parser.add_argument('--imgnum', default=200, type=int,
                        help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--beginIMG', default=0, type=int, #43
                        help='begin test img number')

    parser.add_argument('--blocksize', default=2, type=int,
                        help='')   
    parser.add_argument('--budget', default=  1000, type=int,
                        help='Maximum queries f0r the attack')
    parser.add_argument('--deviceid', default="2", type=str,
                        help='attack batch size.')
    
    parser.add_argument('--binaryM', default=1, type=int,
                        help='binary search mod, mid 0 or median 1.')
    parser.add_argument('--early_stop', default=1, type=int,
                        help='early_stop')
    parser.add_argument('--initDir', default=-1e5, type=int,
                        help='initial direction, 1,-1,and 0 for random;-1e5 for freqSearch init')
    parser.add_argument('--channels', default=3, type=int,
                        help='output channels, 3 for max channels, 1 for 1 channel for all datas')
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
                        help='')    
    parser.add_argument('--mu', default=0, type=float,
                        help='')   
    parser.add_argument('--initystd', default=0, type=float,
                        help='')  
    parser.add_argument('--initcbstd', default=0, type=float,
                        help='')  
    parser.add_argument('--initcrstd', default=0, type=float,
                        help='')  

    parser.add_argument('--paratype', default=17, type=int,
                        help='')  
    parser.add_argument('--seed', default=0, type=int,
                        help='')  
    parser.add_argument('--saveimg', default=0, type=int,
                        help='')  
    parser.add_argument('--freqratio', default=64, type=int,
                        help='freqratio:1-64')  
    parser.add_argument('--color', default=0, type=int,
                        help='freqratio:1-64')  
    parser.add_argument('--init', default="", type=str,
                        help='"":use variance calculated by other dataset;test:use the ground truth variance for test')  
    args = parser.parse_args()
    savep = f"/data/code/max{args.budget}"
       
    deviceid = args.deviceid
    setSeed(args.seed)
    defense = args.defense
    adbafun = ADBEvaluate(PARA_TYPE=args.paratype)
    os.environ["CUDA_VISIBLE_DEVICES"]=deviceid
    order = 2 if args.norm == 'l2' else np.inf
    print(args)
    torch_model = fetchImageNetModels(args.victimmodel)
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
    imgbase = 'data/imagenet/val'
    from PIL import Image
        
    imagelist = [f for f in os.listdir(imgbase)]
    imagelist.sort(key=mysortkey)

    #for imgpath,labels in testfiles:
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
    global globalblocksize
    globalblocksize = []
    for filename in imagelist:
        imgpath = "{}/{}".format(imgbase,filename)
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
        picture_i = i
       
        original_image, label = xi, yi  # test_dataset[picture_i]
        if orig_correct_picture_num >= args.imgnum:
            break
        if i < args.beginIMG:
            i+=1
            continue
        if torch_model.predict_label(xi) == yi:
            orig_correct_picture_num = orig_correct_picture_num + 1
            
            reres = ATK_ADBA(filename.split(".")[0],torch_model, original_image, i,
                                                                   label, picture_i, args.epsilon, 8,order, None,"",savep,adbafun,args)
            if len(reres) == 12:
                success, que, iter_num, R, R2, avgval, Rline,blacklight_count,blacklight_first_detect,chosenv_list,cossimlist,cossimavgblock_list=reres
            elif  len(reres) == 13:
                success, que, iter_num, R, R2, avgval, Rline,blacklight_count,blacklight_first_detect,chosenv_list,cossimlist,cossimavgblock_list,blocknum=reres
                blocknum_list.append(blocknum)
                continue
           
            RlineQ(Rline, radius_line, args.budget - 1)
            if success == 1 and que <= args.budget:
                atk_success = atk_success + 1
                tot_queries = tot_queries + que
                tot_iters = tot_iters + iter_num
                succ_q_list.append(que)
                succ_l2_list.append(R2)
                succ_linf_list.append(R)
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
       
    print(f"ORIGINAL_CLASSIFY_ACC={orig_correct_picture_num / i}")

if __name__ == "__main__":
    main_ADBA()
