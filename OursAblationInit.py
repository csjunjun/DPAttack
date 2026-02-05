# coding:utf-8
import os
import math
import argparse
import torch
import numpy as np
import copy,pywt
import random
from torchvision import transforms
import sys
from torchvision.utils import save_image
from tools.utils import setSeed,progress_bar
from models.OursClass import Block,V,Iter
import statistics
from tools.fetchmodel import fetchImageNetModels
def calAvgBlockCossim(gtgradsign,ourinit):
    toevaluate = gtgradsign
    i=0
    avgpoolres= []
    cossimlist=[]
    for di in ourinit:
        if i==0:
            initsign = ourinit[0]
            sumii_real=[toevaluate[i]]
            sumii=[initsign]
        else:
            if di == sumii[0]:
                sumii_real.append(toevaluate[i])
                sumii.append(di)
            else:
                avgsumii = np.mean(sumii_real)
                sumii_real=[np.sign(avgsumii) for j in sumii_real]
                avgpoolres.append(sumii_real)
                cossim = torch.cosine_similarity(torch.tensor(avgpoolres[-1]).unsqueeze(0),torch.tensor(gtgradsign[i-len(sumii):i]).unsqueeze(0))
                
                cossimlist.append(float(cossim))
                sumii_real = [toevaluate[i]]
                sumii = [di]

        i+=1
    return np.mean(cossimlist)


def getNewdRays(blocksize=4,h=224,w=224):
    p=0
    sitem = []
    gtgradsign=[1 for i in range(3*h*w)]
    gtgradsign_avgpp=[]
    flag=1
    for gi in gtgradsign:
        if (p+1)%blocksize==0:
            if p==0:
                sitem=[flag]
            else:
                sitem.append(flag)
                gtgradsign_avgpp.append([j for j in sitem])

        else:
            if p%blocksize==0:
                flag *=-1
                sitem=[flag]
            else:
                sitem.append(flag)
        p+=1
    gtgradsign_avgppflatten = torch.tensor(np.array(gtgradsign_avgpp)).reshape(1,3,224,224)
    return gtgradsign_avgppflatten



@torch.no_grad()
def ATK_ADBA(filename,model, original_image_x, img_number, label_y, sample_index, aim_r, tolerance_binary_iters,\
              order,tracker,initvariables,savep,adbafun,args,random_targetimg_tensor=None):
    channels, size_x, size_y = original_image_x.shape[1], original_image_x.shape[2], original_image_x.shape[3]
    gtgrad=None
    nonzeroidx=None
    cossimlist_init=None
    method =  ""
    pix_num = channels * size_x * size_y
    npop = 1 
    blocksize=args.blocksize
    print("blocksize:",blocksize) 
    if args.ablation == 5 or args.ablation == 51:#d_r;
        hshape,wshape =original_image_x.shape[2],original_image_x.shape[3]
        randomint_tmp = np.random.randint(0,256,(npop,3,hshape//blocksize,wshape//blocksize) )
            
        randomint = torch.tensor(randomint_tmp).unsqueeze(4).unsqueeze(5).repeat(1,1,1,1,blocksize,blocksize)
        randomint = randomint.permute(0,1,2,4,3,5).flatten(start_dim=-2)
        randomint = randomint.permute(0,1,4,2,3).flatten(start_dim=-2)
        randomint = randomint/255
        randomint = torch.clamp(randomint,0,1).cuda()
        perturbimg = 0.5*randomint+0.5*original_image_x.cuda()
        perturbimg = torch.clamp(perturbimg,0,1)
        newd = perturbimg-original_image_x.cuda()
    elif args.ablation == 6 or args.ablation == 61:#6:anotherimage
        perturbimg = 0.5*random_targetimg_tensor.cuda()+0.5*original_image_x.cuda()
        perturbimg = torch.clamp(perturbimg,0,1)
        newd = perturbimg-original_image_x.cuda()
    elif args.ablation == 4 or args.ablation == 41:#d_b
        newd = getNewdRays(blocksize=blocksize,h=original_image_x.shape[2],w=original_image_x.shape[3])
        newd = newd.float()
    elif args.ablation == 3 or args.ablation == 31:#GaussianInit
        perturbimg = 0.5*torch.randn_like(original_image_x).cuda()+0.5*original_image_x.cuda()
        perturbimg = torch.clamp(perturbimg,0,1)
        newd = perturbimg-original_image_x.cuda()
    elif args.ablation == 2 or args.ablation == 21:#UniformInit
        perturbimg = 0.5*torch.rand_like(original_image_x).cuda()+0.5*original_image_x.cuda()
        perturbimg = torch.clamp(perturbimg,0,1)
        newd = perturbimg-original_image_x.cuda()
    adv_v_init = list(np.array(torch.sign(newd[0]).flatten().cpu().numpy(),dtype=np.int32))
    v0 = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=adv_v_init )
    newd = newd/torch.norm(newd)
    zeronum = len(torch.where(torch.sign(newd)==0)[0])
    print(f"newd_low, zeronum:{zeronum}")
    if zeronum>0:
        newd[torch.where(newd==0)]=int(args.zerosign)
        sign_new_low = torch.sign(newd)
        sign_new_low[torch.where(sign_new_low==0)] = int(args.zerosign)
        tmp = np.array(sign_new_low[0].detach().flatten().cpu().numpy(),dtype=np.int32)
    else:
        tmp = np.array(torch.sign(newd[0]).detach().flatten().cpu().numpy(),dtype=np.int32)
    v0 = V(args.ablation,channels, size_x, size_y, args.initDir,adv_v=tmp )

   
    newd_image = v0.advv_to_tensor()
    newd_image_binary = newd_image
    

    query = 0
    initrhigh=1
    initrlow=0
    success = -1
    candi = None
    chosenv_list = []
    thislimit = min(10,args.budget)
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
    tmp = torch.clamp(original_image_x+initrhigh*newd_image_binary,0,1)
    dis = torch.norm(tmp-original_image_x,p=np.inf)

    candi = tmp

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

            if args.ablation==0 or (args.ablation>1 and args.ablation <10):
                bloc_num = len(v0.continue_subarr)
                b0 = Block(0, bloc_num - 1)
                bs1 = b0.cut_block(args.offspringN)
                blocks = [bs1]
            elif args.ablation==1 or args.ablation >10:
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
            while (query < args.budget) and (ITERATION.old_vbest.Rmax > aim_r):  #main ATK loop

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
                        help='Dataset,DeiTB')
    parser.add_argument('--datasource', default='imagenet', type=str,
                        help='imagenet,imagenetc')
    parser.add_argument('--ablation', default=4, type=int,
                        help='2:UniformInit+PDO,3,GaussianInit+PDO,4:d_b+PDO,5:d_r+PDO,6:anotherimage+PDO\
                            21:UniformInit+ADBAsearch,31,GaussianInit+ADBAsearch,41:d_b+ADBAsearch,51:d_r+ADBAsearch,61:anotherimage+ADBAsearch')


    parser.add_argument('--zerosign', default=1, type=int,
                        help='')

    parser.add_argument('--replace', default=0, type=int,
                        help='Dataset')
    parser.add_argument('--useadba', default=1, type=int,
                        help='Dataset')
    parser.add_argument('--binaryAnalyze', default=0, type=int,
                        help='Dataset')
    parser.add_argument('--onlyone', default=1, type=int,
                        help='0::both,1:only std sample;2:only low color square;.')
    parser.add_argument('--lowtype', default="rcolor", type=str,
                        help='dct,bar,rcolor,std') 
    parser.add_argument('--dwtlevel', default=4, type=float,
                        help='0:no;1;2')  
    parser.add_argument('--dctTrunc', default=4, type=int,
                        help='0:no;1;2')  
    parser.add_argument('--lowfilter', default=0, type=int,
                        help='0:dwt;1:bdct')  

    parser.add_argument('--epsilon', default=0.05, type=float,
                        help='attack strength')
    parser.add_argument('--gradnpop', default=10, type=int,
                        help='attack strength')
    parser.add_argument('--imgnum', default=200, type=int,
                        help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--beginIMG', default=0, type=int, #43
                        help='begin test img number')

    parser.add_argument('--freqratio', default=4, type=int,
                        help='freqratio:1-64')  

    parser.add_argument('--blocksize', default=64, type=int,
                        help='')   
    parser.add_argument('--budget', default=1000, type=int,
                        help='Maximum queries f0r the attack')
    parser.add_argument('--deviceid', default="0", type=str,
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
    parser.add_argument('--color', default=0, type=int,
                        help='freqratio:1-64')  
    parser.add_argument('--init', default="", type=str,
                        help='"":use variance calculated by other dataset;test:use the ground truth variance for test')  
    args = parser.parse_args()
    savep = f"../data/{args.victimmodel}_bs{args.blocksize}_max{args.budget}"
   
    deviceid = args.deviceid
    setSeed(args.seed)
    defense = args.defense

    os.environ["CUDA_VISIBLE_DEVICES"]=deviceid
    order = 2 if args.norm == 'l2' else np.inf
    print(args)
    torch_model= fetchImageNetModels(args.victimmodel)
   
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
    if args.datasource =="imagenetc":
       
        imgbase = '../data/imagenetc'
    elif args.datasource =="imagenet":
        imgbase = '../data/imagenet/val'
    trans = transforms.Compose([
        transforms.Resize((224,224)),
            transforms.ToTensor()
            ])

    from PIL import Image
    if args.datasource =="imagenetc":
        with open(os.path.join(imgbase,'random_test.txt'), 'r') as f:
            imagelist = f.readlines()
        #imagelist = imagelist[:args.imgnum*3]
        imagelist = imagelist[:args.imgnum]
    elif args.datasource =="imagenet":
        from PIL import Image
        if args.init == "":
            
            imagelist = [f for f in os.listdir(imgbase)]
            imagelist.sort(key=mysortkey)

    
    #for imgpath,labels in testfiles:
    ground_truth  = open(os.path.join('../data/imagenet/imagenet_test.txt'), 'r').read().split('\n')
    i=0
    ttt=0
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
    imagelist_final2000 = imagelist[-2000:] 
    random_targetimg_tensor = None
    
    for filename in imagelist:
        if args.datasource =="imagenet":
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
            
            if args.ablation ==6 or args.ablation ==61:
                targeti = 0
                targetlabelidx = yi 
                while targetlabelidx==yi:
                    targetitmp= targeti+1
                    random_targetimg = imagelist_final2000[targeti]
                    fname_target,random_targetimg_label = ground_truth[2000+targetitmp].split()
                    while int(fname_target.split(".")[0].split("_")[-1])>int(random_targetimg.split(".")[0].split("_")[-1]):
                        targetitmp = targetitmp-1
                        fname_target,random_targetimg_label = ground_truth[2000+targetitmp].split()
                    while int(fname_target.split(".")[0].split("_")[-1])<int(random_targetimg.split(".")[0].split("_")[-1]):
                        targetitmp += 1
                        fname_target,random_targetimg_label = ground_truth[2000+targetitmp].split()
                    assert fname_target==random_targetimg
                    targetlabelidx = int(random_targetimg_label)
                    if targetlabelidx==yi:
                        targeti +=1
                        continue 
                    else:
                        random_targetimg_tensor = "{}/{}".format(imgbase,random_targetimg)
                        random_targetimg_tensor = Image.open(random_targetimg_tensor)
                        random_targetimg_tensor = random_targetimg_tensor.convert("RGB")
                        random_targetimg_tensor = trans(random_targetimg_tensor).unsqueeze(0)
                        break

            orig_correct_picture_num = orig_correct_picture_num + 1

            reres = ATK_ADBA(filename.split(".")[0],torch_model, original_image, i,
                                                                   label, picture_i, args.epsilon, 8,order, None,"",\
                                                                    savep,None,args,random_targetimg_tensor)
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
                

            print(  f",\tIMG_{picture_i}"
                f",\tSUCC={success}"
                # f",\tRinf={round(R, 3)}"
                # f",\tR2={round(R2,3)}"
                f",\tQue={que}"
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
