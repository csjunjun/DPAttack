# coding:utf-8
import os 
import math
import argparse
import torch,torchvision
import numpy as np
import copy,pywt


from  ..tools.jpegdct import DiffJPEG

from ..tools.utils import progress_bar

from OursClass import Block,V,getZigzagMeanStd,getNewd,getzigzagcor,getNewdRays
from models.OursDBSSAM import Iter,predictImg


@torch.no_grad()
def ATK_ADBA(model, original_image_x, label_y,img_number, aim_r, tolerance_binary_iters,budget, \
             get_iou_auto,mask_gt,p,iouThreshold=0.5,\
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
            cas,(cHs,cVs,cDs) = pywt.dwt2(original_image_x.detach().cpu(),'haar')
            others = [cHs,cVs,cDs]
        elif dwtlevel==2:
            cas_0,(cHs_0,cVs_0,cDs_0) = pywt.dwt2(original_image_x.detach().cpu(),'haar')
            cas,(cHs,cVs,cDs) = pywt.dwt2(torch.tensor(cas_0),'haar')
            others = [cas_0,cHs_0,cVs_0,cDs_0,cHs,cVs,cDs]
        elif dwtlevel==3:
            cas_0,(cHs_0,cVs_0,cDs_0) = pywt.dwt2(original_image_x.detach().cpu(),'haar')
            cas_1,(cHs_1,cVs_1,cDs_1) = pywt.dwt2(torch.tensor(cas_0),'haar')
            cas,(cHs,cVs,cDs) = pywt.dwt2(torch.tensor(cas_1),'haar')

            others = [cas_0,cHs_0,cVs_0,cDs_0,cas_1,cHs_1,cVs_1,cDs_1,cHs,cVs,cDs]
        elif dwtlevel==4:
            cas_0,(cHs_0,cVs_0,cDs_0) = pywt.dwt2(original_image_x.detach().cpu(),'haar')
            cas_1,(cHs_1,cVs_1,cDs_1) = pywt.dwt2(torch.tensor(cas_0),'haar')
            cas_2,(cHs_2,cVs_2,cDs_2) = pywt.dwt2(torch.tensor(cas_1),'haar')
            cas,(cHs,cVs,cDs) = pywt.dwt2(torch.tensor(cas_2),'haar')

            others = [cas_0,cHs_0,cVs_0,cDs_0,cas_1,cHs_1,cVs_1,cDs_1,cas_2,cHs_2,cVs_2,cDs_2,cHs,cVs,cDs]

    if args.initDir==-1e5:
        if args.binaryAnalyze==2:
            with torch.enable_grad():
                original_image_x.requires_grad = True
                yc,cb,cr = diffj(original_image_x,forged=False,batch=True)
                ycbcr = torch.cat([yc.unsqueeze(1),cb.unsqueeze(1),cr.unsqueeze(1)],dim=1)
        else:
            yc,cb,cr = diffj(original_image_x.detach().cpu(),forged=False,batch=True)
            ycbcr = torch.cat([yc.unsqueeze(1),cb.unsqueeze(1),cr.unsqueeze(1)],dim=1)
        
        if args.onlyone != 1:
            if args.lowtype == "rcolor":
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

            newd,newcandi,newdis,newd_color = getNewd(ycbcr,n_block,stds,original_image_x.cuda(),npop=npop,nchannel=nchannel,\
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
                
                iou_adv_img = predictImg(model,candi,p,mask_gt,get_iou_auto)

            query+=1
            if args.binaryAnalyze==2  or args.binaryAnalyze==4 or args.binaryAnalyze==-1  or args.binaryAnalyze==5 or args.binaryAnalyze==6  or args.binaryAnalyze==7:
                dis = torch.norm(candi-original_image_x.cuda(),p=np.inf)
            else:
                dis = torch.norm(candi-original_image_x,p=np.inf)
            if iou_adv_img>=iouThreshold:
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
                    iou_adv_img = predictImg(model,candi,p,mask_gt,get_iou_auto)
                    query_1 += 1
                    query += 1
                    dis = torch.norm(candi-original_image_x,p=np.inf)
                    if iou_adv_img>=iouThreshold:
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
                    if iou_adv_img>=iouThreshold:
                        dis = tmp
                else:
                    stop1=True
                if  query_1_low<innerqlimit and (dis_low == disbest or flag_low_stop<2):
                    stop2 = False
                    if  flag_low_stop==0:
                        mid_low = (initrhigh_low+initrlow_low)/2
                    else:
                      
                        mid_low = initrhigh_low-(initrhigh_low-initrlow_low)/5
                    if  args.lowtype=="rcolor" or args.lowtype=="bar":
                        candi_low = torch.clamp(original_image_x+mid_low*newd_image_low.cuda(),0,1)

                    # = torch.argmax(model(candi_low.cuda())).cpu()
                    iou_adv_img = predictImg(model,candi_low,p,mask_gt,get_iou_auto)
                    query_1_low += 1
                    query += 1
                    dis_low = torch.norm(candi_low-original_image_x,p=np.inf)
                    if iou_adv_img>=iouThreshold:
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
                    #if pre_low == label_y:
                    if iou_adv_img>=iouThreshold:
                        dis_low = tmp
                else:
                    stop2 = True
                # if flag_low_stop and flag_stop:
                #     break
                innerloop += 1
                if innerloop >= 4 and (initrhigh_low - initrlow_low <= 0.0002 or initrhigh - initrlow <= 0.0002):
                #if ite >= 10 :
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
        #cossimlist = [cossimlist_init]
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
            iou_adv_img = predictImg(model,candi,p,mask_gt,get_iou_auto)
            return success, query, 0, Rbest, np.linalg.norm(nparray, ord=2),candi,iou_adv_img
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
                             iouThreshold=iouThreshold,get_iou_auto=get_iou_auto,mask_gt=mask_gt,p=p)
        
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
                            #blocks[block_iter - 1].remove(bi)
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
            

            success = 1
            if Rbest > aim_r:
                success = -1
            adv_img = adversarial_image - original_image_x

                
            nparray = np.array(adv_img.cpu()).flatten()
            return success, query, ITERATION.iter_n, Rbest, np.linalg.norm(nparray, ord=2), adversarial_image,ITERATION.miniou

def RlineQ(Rline, radius_line, budget):
    start = 0
    for t in range(len(Rline) - 1):
        for q in range(start, min(Rline[t + 1][0], budget)):
            radius_line[q] = radius_line[q] + Rline[t][1]
            start = Rline[t + 1][0]
    return
