# coding:utf-8
import os
import math
import argparse
import torch
import numpy as np
import copy,open_clip
import csv
import random
from torchvision import transforms
import sys
from datetime import datetime
from ..tools.DataTools import ADBEvaluate
from ..models.ADBAClass import Block,V
from ..models.clipmodel import MyCliptModel
import statistics

    
##################################################################################################
#represent blocks of a picture


#represent Iterations
class Iter:
    def __init__(self, init_vbest, offspringN, iter_n=1,early_stop=False,tracker=None,PARA_TYPE=9,useadba=1):
        self.offspringN = offspringN
        self.iter_n = iter_n
        self.offspringVs = [] #d1 d2
        self.early_stop = early_stop
        self.chosen_v = -1 #
        self.old_vbest = copy.deepcopy(init_vbest) # dbest
        for i in range(offspringN):
            self.offspringVs.append(copy.deepcopy(init_vbest))
            self.offspringVs[i].Rmax, self.offspringVs[i].Rmin = 1.0, 0.0
            #
            #
            #
        self.blacklight_threshold = 25 
        self.blacklight_count = 0
        self.blacklight_first_detect = 0
        self.tracker = tracker
        self.adbafun = ADBEvaluate(PARA_TYPE=PARA_TYPE)
        self.adb_interval = 5 
        self.useadba=useadba
        print(f"self.adb_interval:{self.adb_interval}")
    #create a new generation
    def mutation(self, model, original_image, label, aim_r, tolerance_binary_iters, blocks, binaryM,method,globalq=0):
        query = 0
        for vi in range(self.offspringN):
            self.offspringVs[vi].reverse_v(blocks[vi])
            self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = self.old_vbest.Rmax, 0.0

        query = query + self.compare_directions_usingADB(
            model, original_image, label, aim_r, tolerance_binary_iters, binaryM,method,globalq)

        for vi in range(self.offspringN): #initialize directions
            self.offspringVs[vi].reverse_v(blocks[vi])
        if self.chosen_v >= 0:
            self.old_vbest.reverse_v(blocks[self.chosen_v])
            self.old_vbest.Rmax, self.old_vbest.Rmin = (
                self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)
            for vi in range(self.offspringN):
                self.offspringVs[vi].reverse_v(blocks[self.chosen_v])
                self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = (
                    self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)
        self.iter_n = self.iter_n + 1
        return query
    #
    def compare_directions_usingADB(self, model, original_image, label, aim_r, maxIters,\
                                     binaryM,method,globalq=0,paras=None):
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
            predicted.append(torch.argmax(model(perturbed_images[i]).cpu()))
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
                # if self.early_stop:
                #     break
            else:
                self.offspringVs[i].Rmin = self.old_vbest.Rmax

        if len(succV) == 0:
            self.chosen_v = -1
            return query
        elif len(succV) == 1:
            self.chosen_v = succV[0]
            return query

        low, high = 0, self.old_vbest.Rmax
        for ite in range(0, maxIters): #conmaration loop
            if self.useadba==1:
                ADB = self.adbafun.next_ADB(low, high, aim_r, self.old_vbest.Rmax, binaryM,method=method,paras=paras)
            else:
                ADB = high-(high-low)/self.adb_interval
            succVtemp = copy.deepcopy(succV)
            vi = 0
            while vi < len(succVtemp):
                perturbed_images[succVtemp[vi]] = torch.clamp(
                    original_image.cuda() + ADB * perturbations[succVtemp[vi]], 0.0, 1.0)
                predicted[succVtemp[vi]] = torch.argmax(model(perturbed_images[succVtemp[vi]]).cpu())
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
                    # if self.offspringVs[succVtemp[vi]].Rmax <= aim_r:
                    #     self.chosen_v = succVtemp[vi]
                    #     return query
                    vi = vi + 1
                else:
                    self.offspringVs[succVtemp[vi]].Rmin = ADB
                    succVtemp.pop(vi)

            if len(succVtemp) == 0:
                low = ADB
            elif len(succVtemp) == 1:
                self.chosen_v = succVtemp[0]
                return query
            elif len(succVtemp) >= 2:
                high = ADB
                succV = succVtemp

            if ite >= 4 and high - low <= 0.0002:
                break

        #d1 and d2 are close, just return d1
        self.chosen_v = succV[0]
        return query


@torch.no_grad()
def ATK_ADBA(model, original_image_x, img_number, label_y, sample_index, aim_r, tolerance_binary_iters, order,tracker,args):
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

    query = 0
    Rline = [[0, 1.0]]
    ITERATION = Iter(v0, args.offspringN, 1,early_stop=args.early_stop,tracker=tracker,PARA_TYPE=args.paratype)
    query = query + ITERATION.mutation(model, original_image_x, label_y, aim_r, tolerance_binary_iters, blocks[0],
                                       args.binaryM,method=method,globalq=query)

    #progress_bar(img_number, query, iter_num, ITERATION.old_vbest.Rmax)
    Rline.append([query, ITERATION.old_vbest.Rmax])
    """"""

    while (query < args.budget) and (ITERATION.old_vbest.Rmax > aim_r):  
        block_iter = block_iter + 1
        blocks_i = []
        for i, bi in enumerate(blocks[block_iter - 1]):
            blocks_i.extend(bi.cut_block(args.offspringN))
            query_plus = ITERATION.mutation(model, original_image_x, label_y, aim_r, tolerance_binary_iters,
                                            blocks_i[args.offspringN * i:args.offspringN * (i + 1)], args.binaryM,method=method,globalq=query)
            query = query + query_plus


            #progress_bar(img_number, query, iter_num, ITERATION.old_vbest.Rmax)
            #print(f'Img{img_number} Query{query :.0f}\t Iter{iter_num :.0f}\t Rinf{ITERATION.old_vbest.Rmax:.4f}')
            Rline.append([query, ITERATION.old_vbest.Rmax])
            iter_num = iter_num + 1
            if (ITERATION.old_vbest.Rmax <= aim_r) or query >= args.budget:
                break
        blocks.append(copy.deepcopy(blocks_i))

    Rbest = ITERATION.old_vbest.Rmax
    adversarial_v = ITERATION.old_vbest.advv_to_tensor()
    adversarial_image = original_image_x.cpu() + Rbest * adversarial_v
    adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)

    success = 1
    if Rbest > aim_r:
        success = -1
    adv_img = adversarial_image - original_image_x.cpu()

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
        nparray), Rline ,ITERATION.blacklight_count,ITERATION.blacklight_first_detect # np.linalg.norm(nparray,ord=np.inf)


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
    parser.add_argument('--dataset', default='clip', type=str,
                        help='Dataset')
    parser.add_argument('--epsilon', default=0.05, type=float,
                        help='attack strength')

    parser.add_argument('--budget', default=100, type=int,
                        help='Maximum queries for the attack')
    parser.add_argument('--deviceid', default="1", type=str,
                        help='attack batch size.')

    parser.add_argument('--imgnum', default=200, type=int,
                        help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--beginIMG', default=0, type=int,
                        help='begin test img number')
    parser.add_argument('--binaryM', default=1, type=int,
                        help='binary search mod, mid 0 or median 1.')
    parser.add_argument('--early_stop', default=1, type=int,
                        help='early_stop')
    parser.add_argument('--useadba', default=1, type=int,
                        help='Dataset')

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
    
    os.environ["CUDA_VISIBLE_DEVICES"]=deviceid
    order = 2 if args.norm == 'l2' else np.inf
    print(args)

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


    clipM = 'ViT-H-14-CLIPA-336'#，
    clipdataset='datacomp1b' #
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    torch_model, _, preprocess = open_clip.create_model_and_transforms(clipM, \
                                                                     pretrained=clipdataset,device=device)

    torch_model = MyCliptModel(torch_model,clipM,clipdataset,device)


    trans = transforms.Compose([
        transforms.Resize((336,336)),
            transforms.ToTensor()
            ])
    # ###############################################################################


    with open(f"{torch_model.basepp}/cliptest1000_withoutimagenetclasswithidx.txt","r") as f:
        imagelist = f.read().splitlines()
    imagelist = imagelist[:300]
    from PIL import Image
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
    for imgp in imagelist:
        imgpath,labelname,labelinallwithimagenet = imgp.split(",")
        #imgpath = imgpath.replace("yourpath/","..")

        yi = torch.tensor([int(list(torch_model.classlist).index(labelname))])

        xi = Image.open(imgpath)
        xi = xi.convert("RGB")
        xi = trans(xi).unsqueeze(0).cuda()
        yi = yi.cuda()
        predprob = torch_model(xi.cuda())
        pre = torch.argmax(predprob,dim=1)
        if pre != yi:
            print(f"IMG{picture_i} originally classify wrongly")

            continue            

        picture_i = i
        
        original_image, label = xi, yi  # test_dataset[picture_i]
        xi, yi = xi.cuda(), yi.cuda()
        if orig_correct_picture_num >= args.imgnum:
            break
        if i < args.beginIMG:
            i+=1
            continue

        orig_correct_picture_num = orig_correct_picture_num + 1
        success, que, iter_num, R, R2, avgval, Rline,blacklight_count,blacklight_first_detect = ATK_ADBA(torch_model, original_image, i,
                                                                label, picture_i, args.epsilon, 8,order, None,args)
        print(f"img:{i},success:{success},que:{que}")
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
        if success == 1 and que <= args.budget and defense==1:
            print(f"Blacklight detect:{blacklight_count>0},blacklight_detect_ratio:{blacklight_detect_ratio},blacklight_first_detect:{blacklight_first_detect}")
        if len(blacklight_detect_ratios)>0:
            print(f"Avg. blacklight_detect_ratios:{np.mean(np.array(blacklight_detect_ratios))}")
            print(f"Avg. blacklight_first_detects:{np.mean(np.array(blacklight_first_detects))}")
            print(f"Median. blacklight_detect_ratios:{np.median(np.array(blacklight_detect_ratios))}")
            print(f"Median. blacklight_first_detects:{np.median(np.array(blacklight_first_detects))}")
            print(f"Blacklight detect image-wise ratio:{blacklight_succ_detection/(blacklight_succ_detection+blacklight_fail_detection)}")
        i+=1
        
    print(f"ORIGINAL_CLASSIFY_ACC={orig_correct_picture_num / i}")
    

if __name__ == "__main__":
    
    main_ADBA()
