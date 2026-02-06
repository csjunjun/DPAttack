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
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from  tools.jpegdct import DiffJPEG

from tools.DataTools import ADBEvaluate
from tools.utils import setSeed,progress_bar
import statistics
from models.OursClass import *
from apis import MyGoogle,MyImagga,BaiduAPI,TencentAPI


def getTempFilename(image,api_type):
    timpstampnow = int(round(datetime.now().timestamp()))+random.randint(0,1000)
    savep = f"../data/tmp_{api_type}_ours/temp_{timpstampnow}.png"
    Image.fromarray(np.uint8(np.round(image[0].permute(1,2,0).detach().cpu().numpy()*255))).save(savep)

    return savep

@torch.no_grad()
def ATK_ADBA(filename,model, original_image_x, img_number, label_y, sample_index, aim_r, tolerance_binary_iters,\
              order,tracker,initvariables,savep,adbafun,args):
    channels, size_x, size_y = original_image_x.shape[1], original_image_x.shape[2], original_image_x.shape[3]
    if args.channels == 1:
        channels = args.channels
    
    method =  ""
    pix_num = channels * size_x * size_y
    npop = 1 
    nchannel = 3
    step_p = args.stepp
    blocksize=args.blocksize
    print("blocksize:",blocksize)
    diffj = DiffJPEG(bs=blocksize)
    if args.initDir==-1e5:
        yc,cb,cr = diffj(original_image_x.detach().cpu(),forged=False,batch=True)
        ycbcr = torch.cat([yc.unsqueeze(1),cb.unsqueeze(1),cr.unsqueeze(1)],dim=1)
        if args.lowtype == "dct":
            n_block = ycbcr.shape[2]
            if blocksize == 4:
                lowendf = 4
            elif blocksize ==8:
                lowendf = 16

            ldct = torch.zeros_like(ycbcr)
            mdct = torch.zeros_like(ycbcr)
            hdct = torch.zeros_like(ycbcr)

            for lowi in range(0,lowendf):
                idxi,idxj = getzigzagcor(lowi,rows=blocksize,columns=blocksize)
                ldct[:,:,:,idxi,idxj] = ycbcr[:,:,:,idxi,idxj] 

            lowpassimg = diffj.rec(ldct[:,0],ldct[:,1],ldct[:,2],original_image_x.shape[2],original_image_x.shape[3])
            lowpassimg = torch.clamp(lowpassimg,0,1)

        elif args.lowtype == "dwt":
            dwtlevel = args.dwtlevel
            
            
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



            dwtblocksize = 1
            
            #dwtblocksize = int(args.blocksize // (2*dwtlevel))
            print(f"dwtblocksize:{dwtblocksize}")
            cas = torch.tensor(cas)
            global casmin,casmax,casNormalize
            casmin= float(torch.min(cas))
            casmax = float(torch.max(cas))
            casNormalize = (cas-casmin)/(casmax-casmin)

            randomint = np.random.randint(0,256,(npop,3,casNormalize.shape[2]//dwtblocksize,casNormalize.shape[2]//dwtblocksize) )
                
            randomint = torch.tensor(randomint).unsqueeze(4).unsqueeze(5).repeat(1,1,1,1,dwtblocksize,dwtblocksize)
            randomint = randomint.permute(0,1,2,4,3,5).flatten(start_dim=-2)
            randomint = randomint.permute(0,1,4,2,3).flatten(start_dim=-2)
            randomint = randomint/255


            todonp = randomint.detach().cpu().numpy() 

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


        #adv_v_init = list(np.array(torch.sign(newd[0]).flatten().cpu().numpy(),dtype=np.int32))
        v0 = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=adv_v_init )
        if args.lowtype=="dct":
            adv_v_init_color = list(np.array(torch.sign(newd_color[0]).flatten().cpu().numpy(),dtype=np.int32))
            v0_color = V(channels, size_x, size_y, args.initDir,adv_v=adv_v_init_color )





    else:
        v0 = V(channels, size_x, size_y, args.initDir)

    newd_image = v0.advv_to_tensor()
    if args.lowtype=="dct":
        newd_image_color = v0_color.advv_to_tensor()
    elif args.lowtype=="dwt":
        newd_image_low = []
        for v0_low in v0_low_list:
            newd_image_low_tmp = v0_low.advv_to_tensor()
            newd_image_low.append(newd_image_low_tmp)
        newd_image_low = torch.stack(newd_image_low)
    if args.onlyone<=1:
        newd_image_binary = newd_image
    else:
        v0 = v0_low_list[0]
        newd_image_binary = newd_image_low
    if args.budget<=10 or args.onlyone>0:
        query = 0
        initrhigh=1
        initrlow=0
        success = -1
        candi = None
        chosenv_list = []
        while query<10 and query <args.budget:
            mid = (initrhigh+initrlow)/2
            candi = torch.clamp(original_image_x+mid*newd_image_binary,0,1)
            tmpsavep = getTempFilename(candi,args.apitype)
            if args.apitype=="baidu":
                pre = model.detect_labels(tmpsavep,save=False)
            elif args.apitype=="tencent":
                _,pre,_ = model.detect_labels(tmpsavep,query+1,save=False)
            elif args.api_type=="google":
                pre,score =  model.predict_label(tmpsavep,tmpsavep.split(".")[0]+"_res.json")
            elif args.api_type=="imagga":
                pre = model.predict_label(candi)

            query+=1
            dis = torch.norm(candi-original_image_x,p=np.inf)
            if (model.ismatchHard(pre) and (args.api_type=="baidu" or args.api_type=="tencent"))\
                or (args.api_type=="google" and pre == label_y)\
                        or (args.api_type=="imagga" and not model.compare_label(pre)):

                initrlow = mid 
            else:
                initrhigh = mid 
                if dis<=aim_r:
                    success = 1
                    break
        args.paratype=22
        tmp = torch.clamp(original_image_x+initrhigh*newd_image,0,1)
        dis = torch.norm(tmp-original_image_x,p=np.inf)

    else:
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
                candi = torch.clamp(original_image_x+mid*newd_image,0,1)
                tmpsavep = getTempFilename(candi,args.apitype)
                if args.apitype=="baidu":
                    pre = model.detect_labels(tmpsavep,save=False)
                elif args.apitype=="tencent":
                    _,pre,_ = model.detect_labels(tmpsavep,query+1,save=False)
                elif args.api_type=="google":
                    pre,score =  model.predict_label(tmpsavep,tmpsavep.split(".")[0]+"_res.json")
                elif args.api_type=="imagga":
                    pre = model.predict_label(candi)


                query_1 += 1
                query += 1
                dis = torch.norm(candi-original_image_x,p=np.inf)
                if (model.ismatchHard(pre) and (args.api_type=="baidu" or args.api_type=="tencent"))\
                    or (args.api_type=="google" and pre == label_y)\
                         or (args.api_type=="imagga" and not model.compare_label(pre)):

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
                elif args.lowtype=="dwt":
                    candi_low = torch.clamp(original_image_x+mid_low*newd_image_low,0,1)
                tmpsavep = getTempFilename(candi_low,args.apitype)

                if args.apitype=="baidu":
                    pre_low = model.detect_labels(tmpsavep,save=False)
                elif args.apitype=="tencent":
                    _,pre_low,_ = model.detect_labels(tmpsavep,query+1,save=False)
                elif args.api_type=="google":
                    pre_low,score =  model.predict_label(tmpsavep,tmpsavep.split(".")[0]+"_res.json")
                elif args.api_type=="imagga":
                    pre_low = model.predict_label(candi_low)

            
                query_1_low += 1
                query += 1
                dis_low = torch.norm(candi_low-original_image_x,p=np.inf)

                if (model.ismatchHard(pre_low) and (args.api_type=="baidu" or args.api_type=="tencent"))\
                    or (args.api_type=="google" and pre_low == label_y)\
                         or (args.api_type=="imagga" and not model.compare_label(pre_low)):

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
            if innerloop >= 4 and (initrhigh_low - initrlow_low <= 0.0002 or initrhigh - initrlow <= 0.0002):
                print("loop threshold.")
                break
        tmp = torch.clamp(original_image_x+initrhigh*newd_image,0,1)
        if args.lowtype=="dct":
            tmp_low = torch.clamp(lowpassimg+initrhigh_low*newd_image_color,0,1)
        elif args.lowtype=="dwt":
            tmp_low = torch.clamp(original_image_x+initrhigh_low*newd_image_low,0,1)
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
        
    print(f"args.paratype:{args.paratype}")
   

    if args.saveimg==1:
        save_image(tmp,f'{savep}/{img_number}_{int(label_y)}_{query}.png')
        
    if success == 1 or query >= args.budget:
        Rbest=initrhigh
        adv_img = candi-original_image_x
        Rline = [[0, 1.0]]
        nparray = np.array(adv_img.cpu()).flatten()
        return success, query, 0, Rbest, np.linalg.norm(nparray, ord=2), np.mean(
            nparray), Rline ,0,0,chosenv_list #

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
        ITERATION = Iter(v0, args.offspringN, 1,early_stop=args.early_stop,tracker=tracker,paratype=args.paratype,useadba=args.useadba,api_type=args.apitype)
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
                
                query_plus,hisv,hisblock,hispre = ITERATION.mutation(model, original_image_x, label_y, aim_r, tolerance_binary_iters,
                                                blocks_i[args.offspringN * i:args.offspringN * (i + 1)], args.binaryM,method=method,globalq=query,hisv=hisv,hisblock=hisblock,hispre=hispre)
                query = query + query_plus
                Rline.append([query, ITERATION.old_vbest.Rmax])
                if args.saveimg==1:
                    tmp_Rbest = ITERATION.old_vbest.Rmax
                    tmp_adversarial_v = ITERATION.old_vbest.advv_to_tensor()
                    tmp_adversarial_image = original_image_x + tmp_Rbest * tmp_adversarial_v
                    tmp_adversarial_image = torch.clamp(tmp_adversarial_image, 0.0, 1.0)                   
                    save_image(tmp_adversarial_image,f'{savep}/{img_number}_{int(label_y)}_{query}.png')

                iter_num = iter_num + 1
                if (ITERATION.old_vbest.Rmax <= aim_r) or query >= args.budget:
                    break
            if len(blocks_i)>0:
                blocks.append(copy.deepcopy(blocks_i))




        Rbest = ITERATION.old_vbest.Rmax
        adversarial_v = ITERATION.old_vbest.advv_to_tensor()
        adversarial_image = original_image_x + Rbest * adversarial_v
        adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)
        

        success = 1
        if Rbest > aim_r:
            success = -1
        adv_img = adversarial_image - original_image_x

 
            
        nparray = np.array(adv_img.cpu()).flatten()
        return success, query, ITERATION.iter_n, Rbest, np.linalg.norm(nparray, ord=2), np.mean(
            nparray), Rline ,ITERATION.blacklight_count,ITERATION.blacklight_first_detect,chosenv_list # np.linalg.norm(nparray,ord=np.inf)


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
    #profile = lp.LineProfiler()
    # ###################################################################################
    torch_model, test_loader = None, None
    parser = argparse.ArgumentParser(description='Hard Label Attacks')
    parser.add_argument('--apitype', default='baidu', type=str,
                        help='')

    parser.add_argument('--epsilon', default=0.05, type=float,
                        help='attack strength')
    parser.add_argument('--imgnum', default=10, type=int,
                        help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--beginIMG', default=0, type=int, 
                        help='begin test img number')

    parser.add_argument('--blocksize', default=4, type=int,
                        help='')   
    parser.add_argument('--budget', default=50, type=int,
                        help='Maximum queries f0r the attack')
    parser.add_argument('--deviceid', default="3", type=str,
                        help='attack batch size.')
    

    parser.add_argument('--dataset', default='', type=str,
                        help='Dataset')
    parser.add_argument('--replace', default=0, type=int,
                        help='Dataset')
    parser.add_argument('--useadba', default=1, type=int,
                        help='Dataset')
    parser.add_argument('--binaryAnalyze', default=0, type=int,
                        help='Dataset')
    parser.add_argument('--onlyone', default=0, type=int,
                        help='0::both,1:only std sample;2:only low color square.')
    parser.add_argument('--ablation', default=0, type=int,
                        help='0::None,1:adbasearch.')


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
    parser.add_argument('--dwtlevel', default=2, type=float,
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
    parser.add_argument('--lowtype', default="dwt", type=str,
                        help='freqratio:1-64') 
    parser.add_argument('--init', default="", type=str,
                        help='"":use variance calculated by other dataset;test:use the ground truth variance for test')  
    args = parser.parse_args()
      
    deviceid = args.deviceid
    setSeed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"]=deviceid
    order = 2 if args.norm == 'l2' else np.inf
    print(args)

    savep = f"../data/attack_results/{args.apitype}_Ours/"
    if not os.path.exists(savep):
        os.makedirs(savep)
    if args.apitype=="baidu":
        torch_model = BaiduAPI(args.savep)
    elif args.apitype=="tencent":
        torch_model = TencentAPI(args.savep)
    elif args.apitype=="google":
        torch_model = MyGoogle()
    elif args.apitype=="imagga":
        torch_model = MyImagga()
    else:
        print("no such api type.")
        return

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
    
    trans = transforms.Compose([
        transforms.Resize((224,224)),
            transforms.ToTensor()
            ])

    imgbase = '../../data/imagenet/val'
    from PIL import Image

    tgt_text_path = '../classes.txt'
    with open(os.path.join(tgt_text_path), 'r') as f:
        class_text  = f.readlines()
        f.close()
    #for imgpath,labels in testfiles:
    ground_truth  = open(os.path.join('../../data/imagenet/val/imagenet_test.txt'), 'r').read().split('\n')
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
    
    imagelist = [f for f in os.listdir(imgbase)]
    imagelist.sort(key=mysortkey)

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
       
        original_image, label = xi, yi  
        if orig_correct_picture_num >= args.imgnum:
            break
        if i < args.beginIMG:
            i+=1
            continue
        if args.apitype=="baidu":
            oriiresgapath = f"{args.savep}{filename.split('.')[0]}.json"
            if not os.path.exists(oriiresgapath):
                englabel = torch_model.detect_labels(imgpath,save=True)
            else:
                englabel = torch_model.get_labels(oriiresgapath)
            
            torch_model.setgt(englabel)
        elif args.apitype=="tencent":
            oriiresgapath = f"{args.savep}{filename.split('.')[0]}/0.pickle"
            if not os.path.exists(oriiresgapath):
                chlabel,englabel,response = torch_model.detect_labels(imgpath,0,ground_name)
            else:
                chlabel,englabel,response = torch_model.get_labels(oriiresgapath,ground_name)
            
            torch_model.setgt(class_text[yi.item()].strip())
            ismatch = torch_model.ismatchHard(englabel)
            if not ismatch:
                continue
        elif args.apitype=="google":
            oriiresgapath = f"{args.savep}/{filename.split('.')[0]}_ori.json"
            
            englabel,google_score = torch_model.predict_label(oriimaggapath=imgpath,respath=oriiresgapath)
            torch_model.setClean(englabel,google_score)
        elif args.apitype=="imagga":
            oriimaggapath = f"{args.savep}/{filename}_ori224.npy"
            if not os.path.exists(oriimaggapath):
                englabel,ress = torch_model.predict_label(original_image,return_res=True)
                np.save(oriimaggapath,ress)
            else:
                ress = np.load(oriimaggapath,allow_pickle=True)
                englabel = torch_model.predict_label(original_image,request_res=ress)
            torch_model.setClean(englabel)

        
        orig_correct_picture_num = orig_correct_picture_num + 1

        success, que, iter_num, R, R2, avgval, Rline,blacklight_count,blacklight_first_detect,chosenv_list = ATK_ADBA(filename.split(".")[0],torch_model, original_image, i,
                                                                label, picture_i, args.epsilon, 8,order, None,"",savep,None,args)
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
            
        print(f"img:{i},success:{success},que:{que}")

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


        i+=1
        
    print(f"ORIGINAL_CLASSIFY_ACC={orig_correct_picture_num / i}")
    

if __name__ == "__main__":
    
    main_ADBA()
