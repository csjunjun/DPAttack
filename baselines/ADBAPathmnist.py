# coding:utf-8
import os
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

from ..models.ADBAClass import Block,V,Iter
import statistics

@torch.no_grad()
def ATK_ADBA(filename,model, original_image_x, img_number, label_y, sample_index, aim_r, tolerance_binary_iters, order,tracker,args):
    channels, size_x, size_y = original_image_x.shape[1], original_image_x.shape[2], original_image_x.shape[3]

    method = "ADBA"
    pix_num = channels * size_x * size_y

    v0 = V(channels, size_x, size_y, args.initDir)

    iter_num = 1
    block_iter = 0
    b0 = Block(0, pix_num - 1)
    bs1 = b0.cut_block(args.offspringN)
    blocks = [bs1]
    gtgrad = None
    query = 0
    Rline = [[0, 1.0]]
    ITERATION = Iter(v0, args.offspringN, 1,early_stop=args.early_stop,tracker=tracker,PARA_TYPE=args.paratype)
    
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
                #cossimlist.extend(cossim)
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
    torch_model, test_loader = None, None
    parser = argparse.ArgumentParser(description='Hard Label Attacks')
    parser.add_argument('--dataset', default='Net28', type=str,
                        help='Dataset')
    parser.add_argument('--datasource', default='pathmnist', type=str,
                        help='Dataset')

    parser.add_argument('--epsilon', default=0.01, type=float,
                        help='attack strength')
    parser.add_argument('--budget', default=50, type=int,
                        help='Maximum queries for the attack')
    parser.add_argument('--deviceid', default="0", type=str,
                        help='attack batch size.')
    parser.add_argument('--defense', default=0, type=int,
                        help='0:no defense;1:blacklight')
    parser.add_argument('--adaptive', default=0, type=int,
                        help='0:no defense;1:blacklight')
    parser.add_argument('--binaryAnalyze', default=0, type=int,
                        help='8:compare ce, cw grad similarity;7:get gradient similarity during query')


    parser.add_argument('--imgnum', default=200, type=int,
                        help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--beginIMG', default=0, type=int,
                        help='begin test img number')
    parser.add_argument('--binaryM', default=1, type=int,
                        help='binary search mod, mid 0 or median 1.')
    parser.add_argument('--early_stop', default=1, type=int,
                        help='early_stop')
    parser.add_argument('--initDir', default=1, type=int,
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
                        help='0:no defense;1:blacklight')  
    parser.add_argument('--paratype', default=15, type=int,
                        help='0:no defense;1:blacklight')      
    args = parser.parse_args()
    deviceid = args.deviceid
    defense = args.defense
    import medmnist
    from medmnist import INFO, Evaluator
    data_flag = 'pathmnist'#

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    test_dataset = DataClass(split='test', transform=data_transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    
    os.environ["CUDA_VISIBLE_DEVICES"]=deviceid
    targeted = True if args.targeted == '1' else False
    early_stopping = False if args.early == '0' else True
    order = 2 if args.norm == 'l2' else np.inf
    print(args)
    from ..tools.fetchmodel import fetchPMNIST
    torch_model= fetchPMNIST(args.dataset)

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
    


    i=0
    ttt=0
    succ_q_list = []
    all_q_list = []
    succ_l2_list = []
    all_l2_list = []
    succ_linf_list = []
    all_linf_list = []

    for xi, yi in test_loader:
        with torch.no_grad():
            outputs = torch_model(xi.cuda())
        yi = yi.squeeze().long()
        outputs = outputs.softmax(dim=-1)
        pre = torch.argmax(outputs, dim=-1)
        i+=1
        if pre!=yi:
            continue
      
        picture_i = i
        
        original_image, label = xi, yi  # test_dataset[picture_i]
        xi, yi = xi.cuda(), yi.cuda()
       
        if orig_correct_picture_num >= args.imgnum:
            break
        orig_correct_picture_num+=1

        if i < args.beginIMG:
            i+=1
            continue

        success, que, iter_num, R, R2, avgval, Rline,blacklight_count,blacklight_first_detect,cossimlist = ATK_ADBA("",torch_model, original_image, i,
                                                                    label, picture_i, args.epsilon, 8,order, None,args)

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
       

        
    print(f"ORIGINAL_CLASSIFY_ACC={orig_correct_picture_num / i}")


if __name__ == "__main__":
    
    main_ADBA()
