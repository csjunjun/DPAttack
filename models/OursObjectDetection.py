# coding:utf-8
import os
import math
import torch
import numpy as np
import copy,pywt

import sys
from ..tools.jpegdct import DiffJPEG
from ..objectDetectionAttack import evaluate_image
from tools.utils import progress_bar
from ..tools.DataTools import ADBEvaluate

from OursClass import Block,V,getzigzagcor,getZigzagMeanStd,getNewdRays

##################################################################################################
#represent blocks of a picture

class Iter:
    def __init__(self, init_vbest, offspringN, iter_n=1,early_stop=False,tracker=None,paratype=10,useadba=1,\
                 iouThreshold=0.5,apthresh=0.2,targetMask=None):
        self.offspringN = offspringN
        self.iter_n = iter_n
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
        self.miniou = 1.0
        self.iouThreshold =iouThreshold
        self.apthresh = apthresh 
        self.targetMask =targetMask 
        self.finalprecision=10 
        self.finalrecall=10

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



    def compare_directions_usingADB(self, model, original_image, label, aim_r, \
                                    maxIters, binaryM,method,globalq=0,hisv=[],hisblock=[],\
                                        blocks=[],hispre=[],orioffspringlen=2,gtgrad=None,cossim=[],targetMask=None):
        perturbations = [] #d1 d2
        perturbed_images = [] #= x+ADB*d
        predicted = [] #= F(x+ADB*d)
        query = 0
        succV = []
        self.chosen_v = -1

        for i in range(len(self.offspringVs)):
            perturbations.append(self.offspringVs[i].advv_to_tensor().cuda())
            perturbed_images.append(torch.clamp(original_image.cuda() +
                                                self.old_vbest.Rmax * perturbations[i], 0.0, 1.0))
            precision,recall,ap = predictImg(model,perturbed_images[i],self.targetMask,self.iouThreshold)

            predicted.append(float(ap))

            query = query + 1
            if predicted[i] <self.apthresh: #success
                succV.append(i)
                self.offspringVs[i].Rmax = self.old_vbest.Rmax
                self.chosen_v = i
                self.miniou = float(ap)
                self.finalprecision = float(precision)
                self.finalrecall = float(recall)


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
                precision,recall,ap = predictImg(model,perturbed_images[succVtemp[vi]],self.targetMask,self.iouThreshold)


                predicted[succVtemp[vi]] = float(ap)

                query = query + 1
                if predicted[succVtemp[vi]]  <self.apthresh:
                    self.offspringVs[succVtemp[vi]].Rmax = ADB
                    self.chosen_v = succVtemp[vi]
                    self.miniou = float(ap)
                    self.finalprecision = float(precision)
                    self.finalrecall = float(recall)

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
            #if ite >= 10 :
                break

        #d1 and d2 are close, just return d1
        self.chosen_v = succV[0] 
        if gtgrad is not None:
            cossim=[float(torch.cosine_similarity(torch.tensor(self.offspringVs[self.chosen_v].adv_v).unsqueeze(0),torch.sign(gtgrad).cpu().flatten(start_dim=1)))]

        return query,hisv,hisblock,hispre,cossim


   
def getNewd(ycbcr,n_block,stds,padded,image,npop,nchannel,step_p,diffj,ord,initmu=0,\
            initystd=0,blocksize=8,init="",\
                freqratio=1,color=0):
    
    y_std,cb_std,cr_std = stds[0],stds[1],stds[2]
    

    n_block = int((padded.shape[2]/blocksize)*(padded.shape[3]/blocksize))
    mu = torch.zeros(1,3,n_block,blocksize,blocksize)
    mu[:,:,:,0,0] += initmu
    modifys = torch.zeros(npop,nchannel,n_block,blocksize,blocksize)

    repeat_num = int(n_block//n_block)
    color=0
    
    step_p_y = 1 #[block_size:1:1e4;2: 100;4:1;8:0.01]
    #for color
    step_p_cb_color = step_p*100 
    step_p_cr_color = step_p*100
    if color:
        step_p_cb = step_p*100 
        step_p_cr = step_p*100
    else:
        step_p_cb = 0
        step_p_cr = 0
    #perturb_range = int(min(freqratio,blocksize**2))
    perturb_range = blocksize**2
    
    print(f"perturb_channel end at:{perturb_range},step_p_y:{step_p_y},step_p_cb:{step_p_cb},step_p_cr:{step_p_cr}")
    for i in range(perturb_range):
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

        #mu_zs[:,:,:,zigzagi,zigzagj] = mu_z
    modifys[:,0,:,:,:] *= step_p_y
    
    modifys_color = modifys.clone()

    modifys[:,1,:,:,:] *= step_p_cb
    modifys[:,2,:,:,:] *= step_p_cr

    modifys_color[:,1,:,:,:] *= step_p_cb_color
    modifys_color[:,2,:,:,:] *= step_p_cr_color
    
    ycbcr_pert = ycbcr.clone()+ modifys.repeat(1,1,repeat_num,1,1)
    
    adv_images = diffj.rec(ycbcr_pert[:,0],ycbcr_pert[:,1],ycbcr_pert[:,2],padded.shape[2],padded.shape[3]) 

    adv_images = adv_images.cuda()
    
    adv_images = torch.clamp(adv_images,0,1)  
    adv_images = adv_images[:,:,:image.shape[2],:image.shape[3]]
    perturb_pixel =  adv_images-image 
    dis = torch.norm(perturb_pixel,p=ord)
    newd = perturb_pixel/torch.norm(perturb_pixel)


    ycbcr_pert_color = ycbcr.clone()+ modifys_color.repeat(1,1,repeat_num,1,1)        
    adv_images_color = diffj.rec(ycbcr_pert_color[:,0],ycbcr_pert_color[:,1],ycbcr_pert_color[:,2],padded.shape[2],padded.shape[3]) 
    adv_images_color = adv_images_color.cuda()
    adv_images_color = adv_images_color[:,:,:image.shape[2],:image.shape[3]]
    adv_images_color = torch.clamp(adv_images_color,0,1)  
    perturb_pixel_color =  adv_images_color-image 
    dis_color = torch.norm(perturb_pixel_color,p=ord)
    newd_color = perturb_pixel_color/torch.norm(perturb_pixel_color)

    
    return newd,adv_images,dis,newd_color



def predictImg(model,image,target,iouThreshold):
    with torch.no_grad():
        outputs = model([image[0]])
        if outputs[0]["boxes"].shape[0]==0:
            return 0.0,0.0,0.0
        tp, fp, num_gt = evaluate_image(target, outputs[0], iouThreshold)
        sorted_idx = torch.argsort(torch.tensor(outputs[0]["scores"].cpu().tolist()), descending=True)
        tp = torch.tensor(tp)[sorted_idx]
        fp = torch.tensor(fp)[sorted_idx]

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recalls = tp_cum / num_gt
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        # mAP (VOC 11-point)
        ap = 0.0
        for t in torch.linspace(0,1,11):
            p = precisions[recalls >= t].max().item() if (recalls >= t).any() else 0
            ap += p
        ap /= 11

    return precisions[-1].item(), recalls[-1].item(),ap

@torch.no_grad()
def ATK_ADBA(model, padded,original_image_x, label_y,img_number, aim_r, tolerance_binary_iters,budget, \
             iouThreshold=0.5,apthresh=0.1,targetMask=None,\
             args=None):
    channels, size_x, size_y = original_image_x.shape[1], original_image_x.shape[2], original_image_x.shape[3]
    gtgrad=None
    nonzeroidx=None
    cossimlist_init=None
    method =  ""
    pix_num = channels * size_x * size_y
    npop = 1 
    nchannel = 3
    step_p = args.stepp
    blocksize = args.blocksize
    print("blocksize:",blocksize)
    diffj = DiffJPEG(bs=blocksize)
    dwtlevel = int(math.log2(blocksize))
    if dwtlevel>0  and args.lowtype!="dct" and args.onlyone!=1:
        dwtblocksize = 1
        print(f"dwtblocksize:{dwtblocksize}")

        if dwtlevel==1:
            cas,(cHs,cVs,cDs) = pywt.dwt2(padded.detach().cpu(),'haar')
            others = [cHs,cVs,cDs]
        elif dwtlevel==2:
            cas_0,(cHs_0,cVs_0,cDs_0) = pywt.dwt2(padded.detach().cpu(),'haar')
            cas,(cHs,cVs,cDs) = pywt.dwt2(torch.tensor(cas_0),'haar')
            others = [cas_0,cHs_0,cVs_0,cDs_0,cHs,cVs,cDs]
        elif dwtlevel==3:
            cas_0,(cHs_0,cVs_0,cDs_0) = pywt.dwt2(padded.detach().cpu(),'haar')
            cas_1,(cHs_1,cVs_1,cDs_1) = pywt.dwt2(torch.tensor(cas_0),'haar')
            cas,(cHs,cVs,cDs) = pywt.dwt2(torch.tensor(cas_1),'haar')

            others = [cas_0,cHs_0,cVs_0,cDs_0,cas_1,cHs_1,cVs_1,cDs_1,cHs,cVs,cDs]
        elif dwtlevel==4:
            cas_0,(cHs_0,cVs_0,cDs_0) = pywt.dwt2(padded.detach().cpu(),'haar')
            cas_1,(cHs_1,cVs_1,cDs_1) = pywt.dwt2(torch.tensor(cas_0),'haar')
            cas_2,(cHs_2,cVs_2,cDs_2) = pywt.dwt2(torch.tensor(cas_1),'haar')
            cas,(cHs,cVs,cDs) = pywt.dwt2(torch.tensor(cas_2),'haar')

            others = [cas_0,cHs_0,cVs_0,cDs_0,cas_1,cHs_1,cVs_1,cDs_1,cas_2,cHs_2,cVs_2,cDs_2,cHs,cVs,cDs]

    if args.initDir==-1e5:
        yc,cb,cr = diffj(padded.detach().cpu(),forged=False,batch=True)
        ycbcr = torch.cat([yc.unsqueeze(1),cb.unsqueeze(1),cr.unsqueeze(1)],dim=1)
        if args.onlyone != 1:
            if args.lowtype == "dct":
                n_block = ycbcr.shape[2]
                lowendf = args.dctTrunc
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
                    if args.zerosign==0:
                        initsign = np.random.choice([-1,1])
                        newd_low[torch.where(newd_low==0)]=initsign
                        sign_new_low = torch.sign(newd_low)
                        sign_new_low[torch.where(sign_new_low==0)] = initsign 
                    elif args.zerosign==1 or args.zerosign==-1:

                        newd_low[torch.where(newd_low==0)]=int(args.zerosign)
                        sign_new_low = torch.sign(newd_low)
                        sign_new_low[torch.where(sign_new_low==0)] = int(args.zerosign)
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
                    hshape,wshape =padded.shape[2],padded.shape[3]
                    dwtblocksize = args.blocksize
                randomint_tmp = np.random.randint(0,256,(npop,3,wshape//dwtblocksize,hshape//dwtblocksize) )
                    
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

            elif args.lowtype =="std":
                diffj_low = DiffJPEG(bs=dwtblocksize)
            if dwtlevel>0  and args.lowtype!="dct":
                cas = torch.tensor(cas)
                global casmin,casmax,casNormalize
                casmin= float(torch.min(cas))
                casmax = float(torch.max(cas))
                casNormalize = (cas-casmin)/(casmax-casmin)

                cHs_repeat = torch.tensor(cHs).repeat(npop,1,1,1).numpy()
                cVs_repeat = torch.tensor(cVs).repeat(npop,1,1,1).numpy()
                cDs_repeat = torch.tensor(cDs).repeat(npop,1,1,1).numpy()
                if args.lowtype=="std":
                    yc_low,cb_low,cr_low = diffj_low(casNormalize.detach().cpu(),forged=False,batch=True)
                    ycbcr_low = torch.cat([yc_low.unsqueeze(1),cb_low.unsqueeze(1),cr_low.unsqueeze(1)],dim=1)
                    n_block_low = ycbcr_low.shape[2]
                    _,yc_std_low,_ = getZigzagMeanStd(yc_low[0])
                    _,cb_std_low,_ = getZigzagMeanStd(cb_low[0])
                    _,cr_std_low,_ = getZigzagMeanStd(cr_low[0])
                    stds_low =[yc_std_low,cb_std_low,cr_std_low]


                    newd,newcandi,newdis,newd_color = getNewd(ycbcr_low,n_block_low,stds_low,casNormalize.cuda(),npop=npop,nchannel=nchannel,\
                                                step_p=step_p,diffj=diffj_low,ord=np.inf,initmu=args.mu,initystd=args.initystd,\
                                                    blocksize=dwtblocksize,\
                                                        init="",freqratio=args.freqratio,color = args.color) 
                    
                    todonp_unnorm = newcandi.detach().cpu()*(casmax-casmin)+casmin
                else:
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


                recons = torch.clamp(torch.tensor(recons),0,1).cuda()
                recons = recons[:,:,:original_image_x.shape[2],:original_image_x.shape[3]]
                perturb_pixel =  recons-original_image_x.repeat(npop,1,1,1).cuda()
                dis = torch.norm(perturb_pixel,p=np.inf)
                newd_low = perturb_pixel/torch.norm(perturb_pixel)
                zeronum = len(torch.where(torch.sign(perturb_pixel)==0)[0])
                print(f"newd_low, zeronum:{zeronum}")
                if zeronum>0:
                    if args.zerosign==0:
                        initsign = np.random.choice([-1,1])
                        newd_low[torch.where(newd_low==0)]=initsign
                        sign_new_low = torch.sign(newd_low)
                        sign_new_low[torch.where(sign_new_low==0)] = initsign 
                    elif args.zerosign==1 or args.zerosign==-1:

                        newd_low[torch.where(newd_low==0)]=int(args.zerosign)
                        sign_new_low = torch.sign(newd_low)
                        sign_new_low[torch.where(sign_new_low==0)] = int(args.zerosign)
                    tmp = np.array(sign_new_low.detach().flatten(start_dim=1).cpu().numpy(),dtype=np.int32)
                else:
                    tmp = np.array(torch.sign(newd_low).detach().flatten(start_dim=1).cpu().numpy(),dtype=np.int32)
                v0_low_list = []
                for adv_v_init_low in tmp:
                    v0_low = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=adv_v_init_low )
                    v0_low_list.append(v0_low)

        if args.onlyone<2:
            n_block = ycbcr.shape[2]
            y_mu,y_std,_ = getZigzagMeanStd(yc[0])
            #print(list(y_std[0:8]))
            cb_mu,cb_std,_ = getZigzagMeanStd(cb[0])
            cr_mu,cr_std,_ = getZigzagMeanStd(cr[0])
            stds =[y_std,cb_std,cr_std]
            mus = [y_mu,cb_mu,cr_mu]

            newd,newcandi,newdis,newd_color = getNewd(ycbcr,n_block,stds,padded,original_image_x.cuda(),npop=npop,nchannel=nchannel,\
                                        step_p=step_p,diffj=diffj,ord=np.inf,initmu=args.mu,initystd=args.initystd,\
                                            blocksize=args.blocksize,\
                                                init="",freqratio=args.freqratio,color = args.color) 


            zeronum = len(torch.where(torch.sign(newd)==0)[0])
            print(f"newd, zeronum:{zeronum}")
            if zeronum>0:
                if args.zerosign==0:
                    initsign = np.random.choice([-1,1])
                    newd[torch.where(newd==0)]=initsign
                    sign_new_low = torch.sign(newd)
                    sign_new_low[torch.where(sign_new_low==0)] = initsign 
                elif args.zerosign==1 or args.zerosign==-1:

                    newd[torch.where(newd==0)]=int(args.zerosign)
                    sign_new_low = torch.sign(newd)
                    sign_new_low[torch.where(sign_new_low==0)] = int(args.zerosign)
                adv_v_init = list(np.array(sign_new_low[0].flatten().detach().cpu().numpy(),dtype=np.int32))
            else:
                adv_v_init = list(np.array(torch.sign(newd[0]).flatten().detach().cpu().numpy(),dtype=np.int32))

            v0 = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=adv_v_init )
            if args.lowtype=="dct":
                adv_v_init_color = list(np.array(torch.sign(newd_color[0]).flatten().cpu().numpy(),dtype=np.int32))
                v0_color = V(channels, size_x, size_y, args.initDir,adv_v=adv_v_init_color )




    else:
        v0 = V(channels, size_x, size_y, args.initDir)

    if args.onlyone!=1:
        newd_image_low = []
        for v0_low in v0_low_list:
            newd_image_low_tmp = v0_low.advv_to_tensor()
            newd_image_low.append(newd_image_low_tmp)
        newd_image_low = torch.stack(newd_image_low)

    if args.onlyone<=1:
        newd_image = v0.advv_to_tensor()
        newd_image_binary = newd_image
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
                candi = torch.clamp(original_image_x+mid*newd_image_binary.cuda(),0,1)
                
                precision,recall,ap = predictImg(model,candi,targetMask,iouThreshold)

            query+=1
            if args.binaryAnalyze==2  or args.binaryAnalyze==4 or args.binaryAnalyze==-1  or args.binaryAnalyze==5 or args.binaryAnalyze==6  or args.binaryAnalyze==7:
                dis = torch.norm(candi-original_image_x.cuda(),p=np.inf)
            else:
                dis = torch.norm(candi-original_image_x,p=np.inf)
            if ap>=apthresh:
                initrlow = mid 
            else:
                initrhigh = mid 
                if dis<=aim_r:
                    success = 1
                    break
        args.paratype=22

        tmp = torch.clamp(original_image_x+initrhigh*newd_image_binary.cuda(),0,1)
        dis = torch.norm(tmp-original_image_x,p=np.inf)




    else:


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
                    candi = torch.clamp(original_image_x+mid*newd_image.cuda(),0,1)
                    precision,recall,ap = predictImg(model,candi,targetMask,iouThreshold)
                    query_1 += 1
                    query += 1
                    dis = torch.norm(candi-original_image_x,p=np.inf)
                    if ap>=apthresh:
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
                    if ap>=apthresh:
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
                    elif args.lowtype=="rcolor" or args.lowtype=="bar":
                        candi_low = torch.clamp(original_image_x+mid_low*newd_image_low.cuda(),0,1)

                    precision,recall,ap = predictImg(model,candi_low,targetMask,iouThreshold)
                    query_1_low += 1
                    query += 1
                    dis_low = torch.norm(candi_low-original_image_x,p=np.inf)
                    if ap>=apthresh:
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
                    if ap>=apthresh:
                        dis_low = tmp
                else:
                    stop2 = True
                innerloop += 1
                if innerloop >= 4 and (initrhigh_low - initrlow_low <= 0.0002 or initrhigh - initrlow <= 0.0002):
                    print("loop threshold.")
                    break
            tmp = torch.clamp(original_image_x+initrhigh*newd_image.cuda(),0,1)
            if  args.lowtype=="rcolor" or args.lowtype=="bar":
                tmp_low = torch.clamp(original_image_x+initrhigh_low*newd_image_low.cuda(),0,1)
            dis = torch.norm(tmp-original_image_x,p=np.inf)
            dis_low = torch.norm(tmp_low-original_image_x,p=np.inf)
            print(f"dis:{float(dis)},dis_low:{float(dis_low)}")


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
                cossimlist_init = float(torch.cosine_similarity(torch.sign(candi.cpu()-original_image_x.cpu()).flatten(start_dim=1).cpu(),torch.sign(gtgrad).flatten(start_dim=1).cpu(),dim=1))
            else:
                cossimlist_init = None

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
            precision,recall,ap = predictImg(model,candi,targetMask,iouThreshold)
            return success, query, 0, Rbest, np.linalg.norm(nparray, ord=2),candi,ap,precision,recall
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
            ITERATION = Iter(v0, args.offspringN, 1,early_stop=args.early_stop,paratype=args.paratype,useadba=args.useadba,\
                             iouThreshold=iouThreshold,apthresh=apthresh,targetMask=targetMask)
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
                                            step_p=step_p,diffj=diffj,ord=np.inf,initmu=args.mu,initystd=args.initystd,\
                                                initcbstd=args.initcbstd,initcrstd=args.initcrstd,blocksize=args.blocksize,\
                                                    init=args.init,initvariables=[],freqratio=args.freqratio,color = args.color) 
                
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


                while (query < args.budget) and (ITERATION.old_vbest.Rmax > aim_r):  

                    block_iter = block_iter + 1
                    blocks_i = []
                    for i, bi in enumerate(blocks[block_iter - 1]):
                        updated = False
                        if bi.x2-bi.x1==0:
                            if bi.x2==len(ITERATION.old_vbest.continue_subarr):
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
            adversarial_image = original_image_x + Rbest * adversarial_v.cuda()
            adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)
            if ITERATION.finalprecision==10 or ITERATION.finalrecall==10:
                precision,recall,ap = predictImg(model,adversarial_image,targetMask,iouThreshold)
                print(f"ITERATION.finalprecision:{ITERATION.finalprecision},ITERATION.finalrecall:{ITERATION.finalrecall}")
                ITERATION.finalprecision = float(precision)
                ITERATION.finalrecall = float(recall)
            success = 1
            if Rbest > aim_r:
                success = -1
            adv_img = adversarial_image - original_image_x

                
            nparray = np.array(adv_img.cpu()).flatten()
            return success, query, ITERATION.iter_n, Rbest, np.linalg.norm(nparray, ord=2), \
                adversarial_image,ITERATION.miniou,ITERATION.finalprecision,ITERATION.finalrecall



