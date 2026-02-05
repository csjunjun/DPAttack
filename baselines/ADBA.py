# coding:utf-8
import os
import argparse
import torch
import numpy as np
import copy
import csv
import random
from torchvision import transforms
import sys
from datetime import datetime



import statistics

from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(root_path))
from tools.fetchmodel import fetchImageNetModels,featchCifar10Models

from tools.utils import progress_bar,get_tracker

from models.ADBAClass import *

    

@torch.no_grad()
def ATK_ADBA(filename,model, original_image_x, img_number, label_y, sample_index, aim_r, tolerance_binary_iters, order,tracker,args):
    channels, size_x, size_y = original_image_x.shape[1], original_image_x.shape[2], original_image_x.shape[3]
    

    if args.channels == 1:
        channels = args.channels
    method ="ADBA"
    pix_num = channels * size_x * size_y

    
    v0 = V(channels, size_x, size_y, args.initDir)

    iter_num = 1
    block_iter = 0
    b0 = Block(0, pix_num - 1)
    bs1 = b0.cut_block(args.offspringN)
    blocks = [bs1]
    gtgrad = None
    if args.binaryAnalyze==7:
        
        gtgradpath = f"../FGSM_CE_origianlx/{filename.split('.')[0]}.pth"            
        
        gtfolder=f"../GradSignSimilarity/FGSM_CE_origianlx_{args.victimmodel}"
        gtgradpath = f"{gtfolder}/{filename.split('.')[0]}.pth"
        if not os.path.exists(gtfolder):
            os.makedirs(gtfolder)
        celos = torch.nn.CrossEntropyLoss(reduction='none')
        if not os.path.exists(gtgradpath):
            print(f"gtpath:{gtgradpath} not exists, calculate grad sign")
            # tmp.requires_grad = True
            with torch.enable_grad():
                
                img = original_image_x.detach().cuda()
                img.requires_grad = True
                output = model(img)
                cost=celos(output, label_y.unsqueeze(0).cuda())
                gtgrad = torch.autograd.grad(cost, img,
                                        retain_graph=False, create_graph=False)[0]
                print(f"grad sign of {filename}:len(0):{len(torch.where(torch.sign(gtgrad)==0)[0])},len(1):{len(torch.where(torch.sign(gtgrad)==1)[0])},len(-1):{len(torch.where(torch.sign(gtgrad)==-1)[0])}")
                
                torch.save(gtgrad.detach().cpu().numpy(),gtgradpath)

        else:
            gtgrad= torch.tensor(torch.load(gtgradpath))

        print(f"gtgradpath:{gtgradpath}")

    query = 0
    Rline = [[0, 1.0]]
    ITERATION = Iter(v0, args.offspringN, 1,early_stop=args.early_stop,tracker=tracker,paratype=args.paratype,api_type=args.api_type)
    
    queryplus,cossim = ITERATION.mutation(model, original_image_x, label_y, aim_r, tolerance_binary_iters, blocks[0],
                                       args.binaryM,method=method,globalq=query,adaptive=args.adaptive,gtgrad=gtgrad,cossim=[])
    if len(cossim)>0:
        cossimlist = [[cossim[0],queryplus]]
    else:
        cossimlist = []
    query = query + queryplus

    
    Rline.append([query, ITERATION.old_vbest.Rmax])
    """"""

        
    while (query < args.budget) and (ITERATION.old_vbest.Rmax > aim_r):  
        block_iter = block_iter + 1
        blocks_i = []
        for i, bi in enumerate(blocks[block_iter - 1]):
            blocks_i.extend(bi.cut_block(args.offspringN))
            
            query_plus,cossim = ITERATION.mutation(model, original_image_x, label_y, aim_r, tolerance_binary_iters,
                                            blocks_i[args.offspringN * i:args.offspringN * (i + 1)], args.binaryM,method=method,globalq=query,adaptive=args.adaptive,gtgrad=gtgrad,cossim=cossim)
            query = query + query_plus
            if len(cossim)>0:
                cossimlist.append([cossim[0],query])

            Rline.append([query, ITERATION.old_vbest.Rmax])
            iter_num = iter_num + 1
            if (ITERATION.old_vbest.Rmax <= aim_r) or query >= args.budget:
                break
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
        nparray), Rline ,ITERATION.blacklight_count,ITERATION.blacklight_first_detect,cossimlist # np.linalg.norm(nparray,ord=np.inf)


def RlineQ(Rline, radius_line, budget):
    start = 0
    for t in range(len(Rline) - 1):
        for q in range(start, min(Rline[t + 1][0], budget)):
            radius_line[q] = radius_line[q] + Rline[t][1]
            start = Rline[t + 1][0]
    return

def mysortkey(filename:str):
    return int(filename.split("_")[2].split(".")[0])  

def main_ADBA():
    #profile = lp.LineProfiler()
    # ###################################################################################
    torch_model = None, None
    parser = argparse.ArgumentParser(description='Hard Label Attacks')
    parser.add_argument('--victimmodel', default='resnet50', type=str,
                        help='for imagenet:resnet50,vit,inv3,dense121,deit,regnet,SwinV2T,ConvNextBase,efficient,wrs50;for imagenetc:HMany,NoisyMix')
    parser.add_argument('--datasource', default='imagenet', type=str,
                        help='imagenet,imagenetc,cifar10')
    parser.add_argument('--epsilon', default=0.05, type=float,
                        help='attack strength')
    parser.add_argument('--budget', default=100, type=int,
                        help='Maximum queries for the attack')
    parser.add_argument('--deviceid', default="1", type=str,
                        help='attack batch size.')
    parser.add_argument('--defense', default=0, type=int,
                        help='0:no defense;1:blacklight')
    parser.add_argument('--adaptive', default=0, type=int,
                        help='0')

    parser.add_argument('--binaryAnalyze', default=0, type=int,
                        help='7:get gradient similarity during query')
    parser.add_argument('--useadba', default=1, type=int,
                        help='useadba para or not')
    parser.add_argument('--imgnum', default=200, type=int,
                        help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--beginIMG', default=0, type=int,
                        help='begin test img number')
    parser.add_argument('--binaryM', default=1, type=int,
                        help='binary search mod, mid 0 or median 1.')
    parser.add_argument('--early_stop', default=1, type=int,
                        help='early_stop')
    parser.add_argument('--initDir', default=1, type=int,
                        help='initial direction, 1,-1,and 0 for random;')
    parser.add_argument('--channels', default=3, type=int,
                        help='output channels, 3 for max channels, 1 for 1 channel for all datas')
    parser.add_argument('--offspringN', default=2, type=int,
                        help='offspring diretion num in new iteration')
    parser.add_argument('--norm', default='np.linf', type=str,
                        help='Norm for attack, linf only')
    parser.add_argument('--batch', default=1, type=int,
                        help='attack batch size.')
    parser.add_argument('--early', default='1', type=str,
                        help='early stopping (stop attack once the adversarial example is found)')
    parser.add_argument('--paratype', default=15, type=int,
                        help='paratype for useadba')      
    args = parser.parse_args()
    deviceid = args.deviceid
    defense = args.defense
    
    os.environ["CUDA_VISIBLE_DEVICES"]=deviceid
    order = 2 if args.norm == 'l2' else np.inf
    print(args)
    if args.datasource == "cifar10":
        torch_model=featchCifar10Models(args.victimmodel)
    elif args.datasource in ["imagenet","imagenetc"]:
        torch_model=fetchImageNetModels(args.victimmodel)
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
    
    if args.datasource in ["imagenet","imagenetc"]:
        trans = transforms.Compose([
            transforms.Resize((224,224)),
                transforms.ToTensor()
                ])
    elif args.datasource =="cifar10":
        trans = transforms.Compose([
                transforms.ToTensor()
                ])
    
    
    from PIL import Image
    if args.datasource =="imagenetc":
        imgbase = '../data/imagenetc'
        with open(os.path.join(imgbase,'imagenetc_test.txt'), 'r') as f:
            imagelist = f.readlines()
        imagelist = imagelist[:args.imgnum*3]

    elif args.datasource =="cifar10":
        imgbase = '../data/cifar10'
        imagelist = [f for f in os.listdir(imgbase)]
        imagelist.sort(key=mysortkey)
        ground_truth  = open(os.path.join('../data/cifar10/val.txt'), 'r').read().split('\n')

    elif args.datasource =="imagenet":
        imgbase = '../data/imagenet'
        imagelist = [f for f in os.listdir(imgbase)]
        imagelist.sort(key=mysortkey)
        ground_truth  = open(os.path.join(imgbase,'imagenet_test.txt'), 'r').read().split('\n')

    i=0
    ttt=0
    succ_q_list = []
    all_q_list = []
    succ_l2_list = []
    all_l2_list = []
    succ_linf_list = []
    all_linf_list = []
    blacklight_succ_detection,blacklight_fail_detection = 0,0
    blacklight_first_detects=[]

    if args.binaryAnalyze>0 or args.binaryAnalyze==-1:
        args.imgnum=1000
        imagelist = imagelist[2000:]
        ttt=2000

    for filename in imagelist:
        if args.datasource in ["imagenet","cifar10"]:

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
            if args.datasource == "imagenet":
                test_secrets_pil = Image.open(imgpath)
                test_secrets_pil = test_secrets_pil.convert("RGB")
                xi = trans(test_secrets_pil).unsqueeze(0)
            elif args.datasource =="cifar10":
                test_secrets_pil = np.load(imgpath)
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
        xi, yi = xi.cuda(), yi.cuda()
        if defense==1: #blacklight,input [0,1]
            tracker = get_tracker(xi, window_size= 50, hash_kept= 50, roundto= 50, step_size= 1, workers= 5)
        else:
            tracker = None

        if orig_correct_picture_num >= args.imgnum:
            break
        if i < args.beginIMG:
            i+=1
            continue

        if torch_model.predict_label(xi) == yi:
            orig_correct_picture_num = orig_correct_picture_num + 1

            success, que, iter_num, R, R2, avgval, Rline,blacklight_count,blacklight_first_detect,cossimlist = ATK_ADBA(filename,torch_model, original_image, i,
                                                                    label, picture_i, args.epsilon, 8,order, tracker,args)
            

            RlineQ(Rline, radius_line, args.budget - 1)

            if len(cossimlist)>0 and args.binaryAnalyze== 7:
                if not os.path.exists(f"../GradSignSimilarity/FGSMGrad/ADBA_ce_{args.victimmodel}"):
                    os.makedirs(f"../GradSignSimilarity/FGSMGrad/ADBA_ce_{args.victimmodel}")
                    print(f"create dir:../GradSignSimilarity/FGSMGrad/ADBA_ce_{args.victimmodel}")
                np.save(f"../GradSignSimilarity/FGSMGrad/ADBA_ce_{args.victimmodel}/{filename.split('.')[0]}.npy",np.array(cossimlist))

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
            if blacklight_count>0:
                
                blacklight_succ_detection += 1
                # blacklight_detect_ratio = blacklight_count/que
                # blacklight_detect_ratios.append(blacklight_detect_ratio)
                blacklight_first_detects.append(blacklight_first_detect)
            else:
                blacklight_fail_detection += 1

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
            print(f"blacklight_succ_detection={blacklight_succ_detection},blacklight_fail_detection={blacklight_fail_detection}")
            print(f"blacklight_detect_ratios={blacklight_succ_detection/(blacklight_succ_detection+blacklight_fail_detection)}")
            print(f"avg blacklight_first_detects={np.mean(np.array(blacklight_first_detects))}")
            print(f"median blacklight_first_detects={np.median(np.array(blacklight_first_detects))}")
            
          

        else:
            print(f"IMG{picture_i} originally classify wrongly")
        i+=1
        
    print(f"ORIGINAL_CLASSIFY_ACC={orig_correct_picture_num / i}")


if __name__ == "__main__":
    
    main_ADBA()
