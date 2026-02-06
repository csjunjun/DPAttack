# coding:utf-8
import os 
import math
import argparse,math
import torch
import numpy as np
import copy,pywt
import random
from torchvision import transforms
import sys
from torchvision.utils import save_image
from  tools.jpegdct import DiffJPEG

from tools.fetchmodel import fetchImageNetModels ,fetchPMNIST   

from tools.utils import setSeed,progress_bar
from tools.DataTools import ADBEvaluate
from models.OursClass import Block,V,getZigzagMeanStd,getNewd,Iter
import statistics





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
    newd_image_binary_list,v0_list = [],[]
    if args.datasource in ["imagenet","imagenetc","objectnet"]:

        blocksize_list = [2,4,8,16]
    elif args.datasource =="pmnist":
        blocksize_list=[2,4,7]
    blocksize_list_ori = blocksize_list
    chosenv_list = []
    newcandi_color_list=[]
    for blocksize in blocksize_list:
        diffj = DiffJPEG(bs=blocksize)
        yc,cb,cr = diffj(original_image_x.detach().cpu(),forged=False,batch=True)
        ycbcr = torch.cat([yc.unsqueeze(1),cb.unsqueeze(1),cr.unsqueeze(1)],dim=1)
        n_block = ycbcr.shape[2]
        y_mu,y_std,_ = getZigzagMeanStd(yc[0])
        cb_mu,cb_std,_ = getZigzagMeanStd(cb[0])
        cr_mu,cr_std,_ = getZigzagMeanStd(cr[0])
        stds =[y_std,cb_std,cr_std]
        mus = [y_mu,cb_mu,cr_mu]
        newd,newcandi,newdis,newd_color,newcandi_color = getNewd(ycbcr,n_block,stds,original_image_x.cuda(),npop=npop,nchannel=nchannel,\
                                    diffj=diffj,ord=order,initmu=args.mu,initystd=args.initystd,\
                                        initcbstd=args.initcbstd,initcrstd=args.initcrstd,blocksize=blocksize,\
                                            init=args.init,initvariables=initvariables,freqratio=args.freqratio,\
                                                color = args.color,returnColorimg=True) 
        newcandi_color_list.append(newcandi_color)

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
        newd_image = v0.advv_to_tensor()
        newd_image_binary = newd_image
        newd_image_binary_list.append(newd_image_binary)
        v0_list.append(v0)

    query = 0
    initrhigh=1
    initrlow=0
    success = -1
    candi = None
    
    thislooplimit = 10
    print(f"thislooplimit:{thislooplimit}")
    thisloop = 0
    rangescale = initrhigh-initrlow
    if args.budget<=20:
        args.earlyexit = 0
    print(f"earlyexit:{args.earlyexit}")

    while thisloop<thislooplimit and query< args.budget:#
        if len(newd_image_binary_list)==1 and args.earlyexit==1:
            if args.onlyone==0:
                break
        mid = (initrhigh+initrlow)/2
        rangescale = initrhigh-initrlow
        succlist,dis_list=[],[]
        for newd_image_binary in newd_image_binary_list:
            candi = torch.clamp(original_image_x+mid*newd_image_binary,0,1)
            pre = torch.argmax(model(candi.cuda())).cpu()
            query+=1
            dis = torch.norm(candi-original_image_x,p=np.inf)
            succlist.append((pre!=label_y).item())
            dis_list.append(float(dis))
        failidx = np.where(np.array(succlist)==False)[0]
        if len(failidx)==len(newd_image_binary_list): 
            initrlow = mid 
        else:
            tmp = [i for num,i in enumerate(newd_image_binary_list) if num not in failidx]
            if len(tmp)==0 and len(newd_image_binary_list)==1:
                break

            newd_image_binary_list = tmp 
            tmp = [i for num,i in enumerate(blocksize_list) if num not in failidx]
            blocksize_list = tmp
            tmp = [i for num,i in enumerate(v0_list) if num not in failidx]
            v0_list = tmp

            
            initrhigh = mid 
            if mid<=aim_r:
                success = 1
                Rbest=initrhigh
                print(f"succ blocksize:{blocksize_list}")
                globalblocksize.extend(blocksize_list)

                candi = torch.clamp(original_image_x+mid*newd_image_binary_list[0],0,1)
                adv_img = candi[0].unsqueeze(0).cuda()-original_image_x.cuda()
                Rline = [[0, 1.0]]
                nparray = np.array(adv_img.cpu()).flatten()
                return success, query, 0, Rbest, np.linalg.norm(nparray, ord=2), np.mean(
                    nparray), Rline ,0,0,[],[],[] # np.linalg.norm(nparray,ord=np.inf)
        thisloop+=1
    if thisloop>=thislooplimit:
        print("warmup loop limit reached")
        print(f"range scale;{rangescale},initrhigh:{initrhigh},initrlow:{initrlow}")

    blocksize = np.random.choice(blocksize_list)
    print(f"blocksize_list:{blocksize_list}")
    choseidx = blocksize_list_ori.index(blocksize)
    newcandi_color = newcandi_color_list[choseidx]
    globalblocksize.extend(blocksize_list)
    usz = np.where(np.array(blocksize_list)==blocksize)[0]

    newd_image_binary = newd_image_binary_list[usz[0]]
    v0 = v0_list[usz[0]]
    print(f"final blocksize:{blocksize},initrhigh:{initrhigh}")
    newd_image = v0.advv_to_tensor()
    newd_image_binary = newd_image
    zeronum = len(torch.where(torch.sign(newd_image_binary)==0)[0])
    print(f"newd, zeronum:{zeronum}")
    if zeronum>0:
        newd_image_binary[torch.where(newd_image_binary==0)]=1
        sign_new_low = torch.sign(newd_image_binary)
        sign_new_low[torch.where(sign_new_low==0)] = 1
        adv_v_init = list(np.array(sign_new_low[0].flatten().detach().cpu().numpy(),dtype=np.int32))
    else:
        adv_v_init = list(np.array(torch.sign(newd_image_binary).flatten().detach().cpu().numpy(),dtype=np.int32))

    v0 = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=adv_v_init )


    dwtlevel = int(math.log2(blocksize))
    if dwtlevel>0 and args.lowtype!="dct" and args.onlyone!=1:
        dwtblocksize = 1

        print(f"dwtblocksize:{dwtblocksize}")

        if dwtlevel==1:
            cas,(cHs,cVs,cDs) = pywt.dwt2(original_image_x.cpu(),'haar')
            others = [cHs,cVs,cDs]
        elif dwtlevel==2:
            cas_0,(cHs_0,cVs_0,cDs_0) = pywt.dwt2(original_image_x.cpu(),'haar')
            cas,(cHs,cVs,cDs) = pywt.dwt2(torch.tensor(cas_0),'haar')
            others = [cas_0,cHs_0,cVs_0,cDs_0,cHs,cVs,cDs]
        elif dwtlevel==3:
            cas_0,(cHs_0,cVs_0,cDs_0) = pywt.dwt2(original_image_x.cpu(),'haar')
            cas_1,(cHs_1,cVs_1,cDs_1) = pywt.dwt2(torch.tensor(cas_0),'haar')
            cas,(cHs,cVs,cDs) = pywt.dwt2(torch.tensor(cas_1),'haar')

            others = [cas_0,cHs_0,cVs_0,cDs_0,cas_1,cHs_1,cVs_1,cDs_1,cHs,cVs,cDs]
        elif dwtlevel==4:
            cas_0,(cHs_0,cVs_0,cDs_0) = pywt.dwt2(original_image_x.cpu(),'haar')
            cas_1,(cHs_1,cVs_1,cDs_1) = pywt.dwt2(torch.tensor(cas_0),'haar')
            cas_2,(cHs_2,cVs_2,cDs_2) = pywt.dwt2(torch.tensor(cas_1),'haar')
            cas,(cHs,cVs,cDs) = pywt.dwt2(torch.tensor(cas_2),'haar')

            others = [cas_0,cHs_0,cVs_0,cDs_0,cas_1,cHs_1,cVs_1,cDs_1,cas_2,cHs_2,cVs_2,cDs_2,cHs,cVs,cDs]

        elif dwtlevel == 5:
            cas_0,(cHs_0,cVs_0,cDs_0) = pywt.dwt2(original_image_x.cpu(),'haar')
            cas_1,(cHs_1,cVs_1,cDs_1) = pywt.dwt2(torch.tensor(cas_0),'haar')
            cas_2,(cHs_2,cVs_2,cDs_2) = pywt.dwt2(torch.tensor(cas_1),'haar')
            cas_3,(cHs_3,cVs_3,cDs_3) = pywt.dwt2(torch.tensor(cas_2),'haar')

            cas,(cHs,cVs,cDs) = pywt.dwt2(torch.tensor(cas_3),'haar')

            others = [cas_0,cHs_0,cVs_0,cDs_0,cas_1,cHs_1,cVs_1,cDs_1,cas_2,cHs_2,cVs_2,cDs_2,cas_3,cHs_3,cVs_3,cDs_3,cHs,cVs,cDs]


        if args.onlyone!=1:
            hshape,wshape = cas.shape[2],cas.shape[3]
            
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

    if args.onlyone!=1:
        newd_image_low = []
        for v0_low in v0_low_list:
            newd_image_low_tmp = v0_low.advv_to_tensor()
            newd_image_low.append(newd_image_low_tmp)
        newd_image_low = torch.stack(newd_image_low)
    if args.onlyone==0:
        with torch.no_grad():
            candi_low = torch.clamp(original_image_x+initrhigh*newd_image_low.cpu(),0,1)
            pre_low = torch.argmax(model(candi_low.cuda())).cpu()
            query += 1
            dis_low = torch.norm(candi_low-original_image_x,p=np.inf)
            if pre_low==label_y:
                candi = torch.clamp(original_image_x+initrhigh*newd_image.cpu(),0,1)
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
                dis,dis_low,disbest = initrhigh,initrhigh,initrhigh
                innerqlimit = tolerance_binary_iters
                print(f"innerqlimit:{innerqlimit}")
                innerloop = 0
                stop1,stop2 = False,False
                prev_dis_low,prev_dis =1,1
                while query<innerqlimit*2 and query <args.budget and success==-1 and abs(query_1-query_1_low)<2 and not stop1 and not stop2:

                    if  query_1_low<innerqlimit and (dis_low == disbest or flag_low_stop<2):
                        stop2 = False
                        mid_low = initrhigh_low-(initrhigh_low-initrlow_low)/5
                        if  args.lowtype=="rcolor" or args.lowtype=="bar" or args.lowtype=="dwtstd":
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

                    innerloop += 1
                    if innerloop >= 4 and (initrhigh_low - initrlow_low <= 0.0002 or initrhigh - initrlow <= 0.0002):
                        print("loop threshold.")
                        break
                tmp = torch.clamp(original_image_x+initrhigh*newd_image.cpu(),0,1)
                if  args.lowtype=="rcolor" or args.lowtype=="bar"  or args.lowtype=="dwtstd":
                    tmp_low = torch.clamp(original_image_x+initrhigh_low*newd_image_low,0,1)
                #
                dis = torch.norm(tmp-original_image_x,p=np.inf)
                dis_low = torch.norm(tmp_low-original_image_x,p=np.inf)
                print(f"dis:{float(dis)},dis_low:{float(dis_low)}")

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
                globalquery+=query 
                query = globalquery

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
                                                    blocks_i[args.offspringN * i:args.offspringN * (i + 1)], args.binaryM,method=method,globalq=query,\
                                                        hisv=hisv,hisblock=hisblock,hispre=hispre,gtgrad=gtgrad,nonzeroidx=nonzeroidx)
                    query = query + query_plus
                    #progress_bar(img_number, query, iter_num, ITERATION.old_vbest.Rmax)
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



            if  (query < args.budget) and (ITERATION.old_vbest.Rmax > aim_r):
                innerloop=0
                diffj = DiffJPEG(bs=blocksize)
                yc,cb,cr = diffj(original_image_x.detach().cpu(),forged=False,batch=True)
                ycbcr = torch.cat([yc.unsqueeze(1),cb.unsqueeze(1),cr.unsqueeze(1)],dim=1)
                n_block = ycbcr.shape[2]
                y_mu,y_std,_ = getZigzagMeanStd(yc[0])
                cb_mu,cb_std,_ = getZigzagMeanStd(cb[0])
                cr_mu,cr_std,_ = getZigzagMeanStd(cr[0])
                stds =[y_std,cb_std,cr_std]
                mus = [y_mu,cb_mu,cr_mu]
                newd_rev,newcandi,newdis,newd_color = getNewd(ycbcr,n_block,stds,original_image_x.cuda(),npop=npop,nchannel=nchannel,\
                                            diffj=diffj,ord=order,initmu=args.mu,initystd=args.initystd,\
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

                        query_plus,hisv,hisblock,hispre = ITERATION.mutation(model, original_image_x, label_y, aim_r, tolerance_binary_iters,
                                                        blocks_i[args.offspringN * i:args.offspringN * (i + 1)], args.binaryM,method=method,globalq=query,hisv=hisv,hisblock=hisblock,hispre=hispre)
  
                        query = query + query_plus
                        #progress_bar(img_number, query, iter_num, ITERATION.old_vbest.Rmax)
                        Rline.append([query, ITERATION.old_vbest.Rmax])
                        iter_num = iter_num + 1
                        if (ITERATION.old_vbest.Rmax <= aim_r) or query >= args.budget:
                            break
                    if len(blocks_i)>0:
                        blocks.append(copy.deepcopy(blocks_i))

                innerloop += 1
            Rbest = ITERATION.old_vbest.Rmax#
            adversarial_v = ITERATION.old_vbest.advv_to_tensor()
            adversarial_image = original_image_x + Rbest * adversarial_v
            adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)
            

            success = 1
            if Rbest > aim_r:
                success = -1
            # nparray = Rbest*np.array(iter_now.vbest.adv_v).flatten()
            adv_img = adversarial_image - original_image_x

            # if query >= 1:
            #     #save and output images and atk images
            #     Filestring = ("Img"+str(sample_index)+
            #                 "_Que"+str(query)+
            #                 "_Time"+str(datetime.now().strftime("%H-%M-%S"))
            #                 )
                
                # combined_file = DataTools.save_images(original_image_x,
                #                                       adversarial_image,
                #                                       0.5 * ITERATION.old_vbest.Rmax * (1 + adversarial_v),
                #                                       Filestring,candi)
                
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


def get_data_iterator(dataset_type,data):
    if dataset_type == 'pmnist':
        for xi, yi in data:
            yield [xi,yi]
            
    else:
        for filename in data:            
            yield filename
def main_ADBA():
    #profile = lp.LineProfiler() --onlyone 1 --blocksize 4 --budget 50 --lowtype rcolor
    # ###################################################################################
    torch_model, test_loader = None, None
    parser = argparse.ArgumentParser(description='Hard Label Attacks')
    parser.add_argument('--victimmodel', default='resnet50', type=str,
                        help='for pmnist:Net28;for objectnet:clip;for imagenet:resnet50,vit,inv3,dense121,deit,regnet,SwinV2T,ConvNextBase,efficient,wrs50;for imagenetc:HMany,NoisyMix')
    parser.add_argument('--datasource', default='imagenet', type=str,
                        help='imagenet,imagenetc,objectnet,pmnist')
    parser.add_argument('--apitype', default='standard', type=str,
                        help='standard:imagenet,imagenetc;clip:objectnet')

    parser.add_argument('--epsilon', default=0.05, type=float,
                        help='attack strength')
    parser.add_argument('--imgnum', default=200, type=int,
                        help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--beginIMG', default=0, type=int, #43
                        help='begin test img number')
 
    parser.add_argument('--budget', default=  100, type=int,
                        help='Maximum queries for the attack')
    parser.add_argument('--deviceid', default="1", type=str,
                        help='')
    
    parser.add_argument('--replace', default=0, type=int,
                        help='Dataset')
    parser.add_argument('--earlyexit', default=0, type=int,
                        help='Dataset')
    parser.add_argument('--warmup', default=0, type=int,
                        help='0: dynamicly choose blocksize for every image; paper default:0\
                            1: warmup to choose the most frequent blocksize of the initial several images;\
                                -1: have warmed up')
    parser.add_argument('--warmupsize', default=20, type=int,
                        help='')


    parser.add_argument('--useadba', default=1, type=int,
                        help='1: use adba paras;0:used line search')
    parser.add_argument('--binaryAnalyze', default=0, type=int,
                        help='default:0 for attack; other values for analysis')
    parser.add_argument('--onlyone', default=0, type=int,
                        help='initial direction:0:use dn and phi(dr), then bilisearch,1:only use dn;2:only use phi(dr).')
    parser.add_argument('--ablation', default=0, type=int,
                        help='0::None,1:adba dyadic search.')
    parser.add_argument('--lowtype', default="rcolor", type=str,
                        help='dct,bar,rcolor,dwtstd;paper default:rcolor') 

    parser.add_argument('--blocksize', default=2, type=int,
                        help='no need to set to do attack;just for analysis')  
    
    parser.add_argument('--binaryM', default=1, type=int,
                        help='binary search mod, mid 0 or median 1.；paper default:1')
    parser.add_argument('--early_stop', default=1, type=int,
                        help='early_stop')
    parser.add_argument('--initDir', default=-1e5, type=int,
                        help='-1e5 for our proposed initialization')
    parser.add_argument('--channels', default=3, type=int,
                        help='output channels, 3 for max channels')
    parser.add_argument('--offspringN', default=2, type=int,
                        help='offspring diretion num in new iteration')

    parser.add_argument('--norm', default='np.linf', type=str,
                        help='Norm for attack, linf only, l2 attack is in another file')
    parser.add_argument('--batch', default=1, type=int,
                        help='attack batch size.')
    parser.add_argument('--early', default='1', type=str,
                        help='early stopping (stop attack once the adversarial example is found)')

    parser.add_argument('--mu', default=0, type=float,
                        help='')   
    parser.add_argument('--initystd', default=0, type=float,
                        help='')  
    parser.add_argument('--initcbstd', default=0, type=float,
                        help='')  
    parser.add_argument('--initcrstd', default=0, type=float,
                        help='')  

    parser.add_argument('--paratype', default=22, type=int,
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
                        help='"":others for analysis')  
    args = parser.parse_args()
    savep = f"{args.victimmodel}_bs{args.blocksize}_max{args.budget}"

      
    deviceid = args.deviceid
    setSeed(args.seed)
    adbafun = ADBEvaluate(PARA_TYPE=args.paratype)
    os.environ["CUDA_VISIBLE_DEVICES"]=deviceid
    order = 2 if args.norm == 'l2' else np.inf
    print(args)
    if args.datasource =="objectnet":
        from models.clipmodel import MyCliptModel
        import open_clip

        clipM = 'ViT-H-14-CLIPA-336'#
        clipdataset='datacomp1b' #
        cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda else "cpu")
        torch_model, _, preprocess = open_clip.create_model_and_transforms(clipM, \
                                                                        pretrained=clipdataset,device=device)

        torch_model = MyCliptModel(torch_model,clipM,clipdataset,device)
        imgsize = 336
        args.apitype='clip'
    elif args.datasource in ["imagenet","imagenetc"]:
        torch_model=fetchImageNetModels(args.victimmodel)
        imgsize = 224
        args.apitype='standard'
    elif args.datasource == "pmnist":
        torch_model=fetchPMNIST(args.victimmodel)
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
        transforms.Resize((imgsize,imgsize)),
            transforms.ToTensor()
            ])
    cossimlist=[]
    if args.datasource =="imagenetc":
        
        imgbase = 'data/imagenetc'
    else:
        imgbase = 'data/imagenet'
    from PIL import Image
    if args.datasource =="imagenetc":
        with open(os.path.join(imgbase,'imagenetc_test.txt'), 'r') as f:
            imagelist = f.readlines()
        imagelist = imagelist[:args.imgnum*3]
        #imagelist = imagelist[:args.imgnum]
    elif args.datasource =="imagenet":
    
        imagelist = [f for f in os.listdir(os.path.join(imgbase,"val"))]
        imagelist.sort(key=mysortkey)
    elif args.datasource =="objectnet":
        with open(f"{torch_model.basepp}/cliptest1000_withoutimagenetclasswithidx.txt","r") as f:
            imagelist = f.read().splitlines()
        imagelist = imagelist[:300]
    elif args.datasource == "pmnist":
        from medmnist import INFO, Evaluator
        data_flag = 'pathmnist'

        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        # preprocessing
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        test_dataset = DataClass(split='test', transform=data_transform, download=False)
        imagelist = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


   
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
    data_iter =get_data_iterator(args.datasource,imagelist)
    #for filename in imagelist:
    for dataitem in data_iter:
        if args.datasource =="imagenet":

            imgpath = "{}/val/{}".format(imgbase,dataitem)
            ground_name_label = ground_truth[ttt]
            ttt+=1 
            
            
            ground_label =  ground_name_label.split()[1]
            ground_name =  ground_name_label.split()[0]
            innerl = 0
            while not dataitem==ground_name:
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
            filename = dataitem
        elif args.datasource =="imagenetc":
            ground_label = dataitem.strip().split(",")[1] 
            yi = torch.tensor(int(ground_label))
            imgpath = dataitem.strip().split(",")[0]
            test_secrets_pil = Image.open(imgpath)
            test_secrets_pil = test_secrets_pil.convert("RGB")
            xi = trans(test_secrets_pil).unsqueeze(0)
            filename = dataitem
        elif args.datasource =="objectnet":
            imgpath,labelname,labelinallwithimagenet = dataitem.split(",")
            #imgpath = imgpath.replace("yourpath/","..")

            yi = torch.tensor([int(list(torch_model.classlist).index(labelname))])
            xi = Image.open(imgpath)
            xi = xi.convert("RGB")
            xi = trans(xi).unsqueeze(0)
            yi = yi.cuda()
            filename = dataitem

        elif args.datasource == "pmnist":
            xi,yi = dataitem
            yi = yi.squeeze().long()
            outputs = torch_model(xi.cuda())
            outputs = outputs.softmax(dim=-1)
            pre = torch.argmax(outputs, dim=-1)
            filename =""




        picture_i = i
       
        original_image, label = xi, yi  # test_dataset[picture_i]
        #xi, yi = xi.cuda(), yi.cuda()
        if orig_correct_picture_num >= args.imgnum:
            break
        if i < args.beginIMG:
            i+=1
            continue
        if (torch_model.predict_label(xi) == yi and args.datasource in ["imagenet","imagenetc"]) or \
            ( torch.argmax(torch_model(xi.cuda()),dim=1) == yi  and args.datasource =="objectnet") or\
                (args.datasource == "pmnist" and pre==yi):
            initvariables = []
            orig_correct_picture_num = orig_correct_picture_num + 1
            reres = ATK_ADBA(filename.split(".")[0],torch_model, original_image, i,
                                                                   label, picture_i, args.epsilon, 8,order, None,"",savep,adbafun,args)
            if orig_correct_picture_num== args.warmupsize and args.warmup==1:
                from collections import Counter
                c = Counter(globalblocksize)
                mostcommonblocksize = c.most_common(1)[0][0]
                print(f"warmup finished,set blocksize to most common size:{mostcommonblocksize}")
                args.blocksize = mostcommonblocksize
                args.warmup=-1 #only warmup once
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
                f",\t Succ Avg.Q={np.mean(np.array(succ_q_list))}"
                f",\t Succ Median.Q={np.median(np.array(succ_q_list))}"
                f",\t Succ Avg.L2={np.mean(np.array(succ_l2_list))}"
                f",\t Succ Median.L2={np.median(np.array(succ_l2_list))}"
                f",\t Succ Avg.Linf={np.mean(np.array(succ_linf_list))}"
                f",\t Succ Median.Linf={np.median(np.array(succ_linf_list))}"

                f",\t All Avg.Q={np.mean(np.array(all_q_list))}"
                f",\t All Median.Q={np.median(np.array(all_q_list))}"
                f",\t All Avg.L2={np.mean(np.array(all_l2_list))}"
                f",\t All Median.L2={np.median(np.array(all_l2_list))}"
                f",\t All Avg.Linf={np.mean(np.array(all_linf_list))}"
                f",\t All Median.Linf={np.median(np.array(all_linf_list))}"

                
            )


        else:
            print(f"IMG{picture_i} originally classify wrongly")
        i+=1
       

    print(f"ORIGINAL_CLASSIFY_ACC={orig_correct_picture_num / i}")
if __name__ == "__main__":
    main_ADBA()
