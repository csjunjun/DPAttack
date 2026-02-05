# coding:utf-8
import os,sys
import argparse
import torch
import numpy as np
import copy
from torchvision import transforms
from datetime import datetime
from torchvision.utils import save_image
import statistics
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(root_path))

from tools.utils import setSeed,progress_bar
from models.ADBAAPIClass import *
from apis import MyGoogle,MyImagga,BaiduAPI,TencentAPI
##################################################################################################

    


@torch.no_grad()
def ATK_ADBA(model, original_image_x, img_number, label_y, sample_index, aim_r, tolerance_binary_iters, \
             order,tracker,filename,args):
    channels, size_x, size_y = original_image_x.shape[1], original_image_x.shape[2], original_image_x.shape[3]
    if args.channels == 1:
        channels = args.channels
    method = "ADBA"
    print(f"real para method:{method}")
    pix_num = channels * size_x * size_y


    v0 = V(channels, size_x, size_y, args.initDir)

    iter_num = 1
    block_iter = 0
    b0 = Block(0, pix_num - 1)
    bs1 = b0.cut_block(args.offspringN)
    blocks = [bs1]

    query = 0
    Rline = [[0, 1.0]]
    ITERATION = Iter(v0, args.offspringN, 1,early_stop=args.early_stop,tracker=tracker,paratype=args.paratype,api_type=args.apitype)

    """"""
    query = query + ITERATION.mutation(model, original_image_x, label_y, aim_r, tolerance_binary_iters, blocks[0],
                                       args.binaryM,method=method,globalq=query)
    progress_bar(img_number, query, iter_num, ITERATION.old_vbest.Rmax)
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
    # nparray = Rbest*np.array(iter_now.vbest.adv_v).flatten()
    adv_img = adversarial_image - original_image_x
    if success == 1:
        sss = f"{args.savep}/{filename.split('.')[0]}.png"
        save_image(adversarial_image, sss)
        print(sss)
    if query >= 1:
        #save and output images and atk images
        Filestring = ("Img"+str(sample_index)+
                     "_Que"+str(query)+
                      "_Time"+str(datetime.now().strftime("%H-%M-%S"))
                      )
        

    nparray = np.array(adv_img.cpu()).flatten()
    return success, query, ITERATION.iter_n, Rbest, np.linalg.norm(nparray, ord=2), np.mean(
        nparray), Rline ,ITERATION.blacklight_count,ITERATION.blacklight_first_detect #


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
    parser.add_argument('--dataset', default='', type=str,
                        help='Dataset')
    parser.add_argument('--apitype', default='baidu', type=str,
                        help='')

    parser.add_argument('--epsilon', default=0.05, type=float,
                        help='attack strength')
    parser.add_argument('--imgnum', default=100, type=int,
                        help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--beginIMG', default=0, type=int,
                        help='begin test img number')
    parser.add_argument('--budget', default=50, type=int,
                        help='Maximum queries for the attack')
    parser.add_argument('--binaryM', default=1, type=int,
                        help='binary search mod, mid 0 or median 1.')
    parser.add_argument('--early_stop', default=1, type=int,
                        help='early_stop')
    parser.add_argument('--initDir', default=1, type=int,
                        help='initial direction, 1,-1,and 0 for random')
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
    parser.add_argument('--deviceid', default="2", type=str,
                        help='attack batch size.')
    parser.add_argument('--paratype', default=15, type=int,
                        help='') 
    parser.add_argument('--savep', default="", type=str,
                        help='attack batch size.')   
    parser.add_argument('--seed', default=0, type=int,
                        help='')     
    args = parser.parse_args()
    setSeed(args.seed)
    deviceid = args.deviceid
    args.savep = f"../data/attack_results/{args.apitype}_ADBA/"
    if not os.path.exists(args.savep):
        os.makedirs(args.savep)
    os.environ["CUDA_VISIBLE_DEVICES"]=deviceid
    order = 2 if args.norm == 'l2' else np.inf
    print(args)
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

    tgt_text_path = '../classes.txt'
    with open(os.path.join(tgt_text_path), 'r') as f:
        class_text  = f.readlines()
        f.close()

    from PIL import Image
    imgbase = '../../data/imagenet/val'
    imagelist = [f for f in os.listdir(imgbase)]
    imagelist.sort(key=mysortkey)

    ground_truth  = open(os.path.join('../../data/imagenet/val/imagenet_test.txt'), 'r').read().split('\n')
    i=0
    ttt=0
    succ_q_list = []
    all_q_list = []
    succ_l2_list = []
    all_l2_list = []
    succ_linf_list = []
    all_linf_list = []
    attacked_img_num=0
    for filename in imagelist:
        imgpath = "{}/{}".format(imgbase,filename)
            
        ground_name_label = ground_truth[ttt]
        ttt+=1 
        
        
        ground_label =  ground_name_label.split()[1]
        ground_name =  ground_name_label.split()[0]
        if not filename==ground_name:
            print("not match.")
        yi = torch.tensor(int(ground_label))
        test_secrets_pil = Image.open(imgpath)
        test_secrets_pil = test_secrets_pil.convert("RGB")
        xi = trans(test_secrets_pil).unsqueeze(0)
        picture_i = i
        
        original_image, label = xi, yi  # 
        xi, yi = xi.cuda(), yi.cuda()
        
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
        success, que, iter_num, R, R2, avgval, Rline,blacklight_count,blacklight_first_detect =\
              ATK_ADBA(torch_model, original_image, i,\
                    englabel, picture_i, args.epsilon, 8,order, None,filename,args)
        RlineQ(Rline, radius_line, args.budget - 1)
        if success == 1 and que <= args.budget:
            atk_success = atk_success + 1
            tot_queries = tot_queries + que
            tot_iters = tot_iters + iter_num
            succ_q_list.append(que)
            succ_l2_list.append(R2)
            succ_linf_list.append(R)
        print(f"img:{i},success:{success},que:{que}")
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
    
        i+=1
        attacked_img_num +=1
    print(f"ORIGINAL_CLASSIFY_ACC={orig_correct_picture_num / i}")


if __name__ == "__main__":
    
    main_ADBA()
