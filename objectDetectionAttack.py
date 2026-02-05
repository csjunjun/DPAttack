import os,math
import torch    
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import argparse
import numpy as np
def box_iou(boxes1, boxes2):
    """
    boxes1: [N,4]  (x1,y1,x2,y2)
    boxes2: [M,4]
    return: IoU [N,M]
    """
    area1 = (boxes1[:,2]-boxes1[:,0]) * (boxes1[:,3]-boxes1[:,1])
    area2 = (boxes2[:,2]-boxes2[:,0]) * (boxes2[:,3]-boxes2[:,1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]
    union = area1[:,None] + area2 - inter
    iou = inter / union
    return iou


def evaluate_image(target, output, iou_thresh=0.5):
    """
    target: dict, 包含 'boxes' [N,4], 'labels' [N]
    output: dict, 包含 'boxes' [M,4], 'labels' [M], 'scores' [M]
    """
    newtarget={"boxes":[],"labels":[]}
    for segres in target:
        x,y,w,h = segres["bbox"]
        newtarget["boxes"].append([x,y,x+w,y+h])
        #newtarget["boxes"].append(segres["bbox"])
        newtarget["labels"].append(segres["category_id"])
    newtarget["boxes"]=torch.tensor(newtarget["boxes"]).float()
    newtarget["labels"]=torch.tensor(newtarget["labels"]).long()
    gt_boxes = newtarget["boxes"].cuda()
    gt_labels = newtarget["labels"].cuda()
    pred_boxes = output["boxes"]
    pred_labels = output["labels"]
    pred_scores = output["scores"]

    idx = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[idx]
    pred_labels = pred_labels[idx]
    pred_scores = pred_scores[idx]

    matched_gt = set()
    tp, fp = [], []

    for i in range(len(pred_boxes)):
        box, label = pred_boxes[i], pred_labels[i]
        ious = box_iou(box.unsqueeze(0), gt_boxes)  # [1, N]
        max_iou, max_j = ious.max(1)
        max_iou = max_iou.item()
        max_j = max_j.item()

        if max_iou >= iou_thresh and label == gt_labels[max_j] and max_j not in matched_gt:
            tp.append(1)
            fp.append(0)
            matched_gt.add(max_j)
        else:
            tp.append(0)
            fp.append(1)

    return tp, fp, len(gt_boxes)



def evaluate_dataset(model, dataloader, device,paradict,args=None):
    all_scores = []
    all_tp = []
    all_fp = []
    total_gts = 0
    attack_method = paradict["attack_method"]
    epsilon = paradict["epsilon"]
    budget = paradict["budget"]
    apthresh = paradict["apthresh"]
    iou_thresh = paradict["iou_thresh"]
    model.eval()
    imgnum = 0
    if attack_method == "ADBA":
        from models.ADBAObjectDetection import ATK_ADBA
    elif attack_method == "Ours":
        from models.OursObjectDetection import ATK_ADBA
    elif attack_method == "Oursdy":
        from models.OursDBSObjectDetection import ATK_ADBA
    imgidx = 0
    querylist,succquerylist = [],[]
    precisionlist,succprecisionlist = [],[]
    recalllist,succrecalllist = [],[]
    aplist,succaplist = [],[]
    with torch.no_grad():
        for images, targets in dataloader:
            toevaluate = []
            inneri =0
            for img in images:
                if attack_method == "ADBA":
                    success, query, _, linfdis, l2dis,adversarial_image,iou_adv_img,finalprecision,finalrecall =\
                    ATK_ADBA(model, img.to(device).unsqueeze(0), None, imgidx, epsilon,8, budget,iouThreshold=iou_thresh,apthresh=apthresh,targetMask =targets[inneri])
                    toevaluate.append(adversarial_image)
                elif attack_method=="Ours" or  attack_method == "Oursdy":
                    c,h,w = img.shape[0],img.shape[1],img.shape[2]
                    pad_h,pad_w = 0,0
                    if args.onlyone<=1:
                        todiv = args.blocksize
                    elif args.onlyone==2:
                        dwtblocksize = int(args.blocksize // (2**args.dwtlevel))

                        todiv = (2**args.dwtlevel)*dwtblocksize
                    if h%todiv!=0:        #padding 0 
                        pad_h = (todiv - (h % todiv)) % todiv  # padding needed on bottom
                    if w%todiv!=0:        #padding 0 
                        pad_w = (todiv - (w % todiv)) % todiv  # padding needed on bottom

                    padded = torch.nn.functional.pad(img.unsqueeze(0), (0,pad_w,0,pad_h), mode="replicate")
                    if attack_method=="Ours":
                        success, query, _, linfdis, l2dis,adversarial_image,iou_adv_img,finalprecision,finalrecall =\
                        ATK_ADBA(model, padded,img.to(device).unsqueeze(0), None, imgidx, epsilon,8, budget,iouThreshold=iou_thresh,apthresh=apthresh,targetMask =targets[inneri],args=args)
                    else:
                         
                        success, query, _, linfdis, l2dis,adversarial_image,_,_,_,_,_,_,iou_adv_img,finalprecision,finalrecall =\
                        ATK_ADBA(model,img.to(device).unsqueeze(0), None, imgidx, epsilon,8, np.inf,iouThreshold=iou_thresh,apthresh=apthresh,targetMask =targets[inneri],args=args)
                    toevaluate.append(adversarial_image)
                elif attack_method == "None":
                    toevaluate = [img.to(device) for img in images]
                imgidx += 1
                inneri += 1
            # outputs = model(toevaluate)

            # for target, output in zip(targets, outputs):
            #     tp, fp, num_gt = evaluate_image(target, output, iou_thresh)
            #     total_gts += num_gt
            #     all_tp.extend(tp)
            #     all_fp.extend(fp)
            #     all_scores.extend(output["scores"].cpu().tolist())
            imgnum += len(toevaluate)
            if attack_method != "None":
                print(f"imgnum:{imgnum},success:{success},query:{query},linfdis:{linfdis},l2dis:{l2dis},ap_adv_img:{iou_adv_img},finalprecision:{finalprecision},finalrecall:{finalrecall}  ")
                querylist.append(query)
                precisionlist.append(finalprecision)
                recalllist.append(finalrecall)
                aplist.append(iou_adv_img)
                if success==1:
                    succquerylist.append(query)
                    succprecisionlist.append(finalprecision)
                    succrecalllist.append(finalrecall)
                    succaplist.append(iou_adv_img)

            if imgnum >= 200:
                break
    if attack_method=="clean":
        sorted_idx = torch.argsort(torch.tensor(all_scores), descending=True)
        tp = torch.tensor(all_tp)[sorted_idx]
        fp = torch.tensor(all_fp)[sorted_idx]

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recalls = tp_cum / total_gts
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        # mAP (VOC 11-point)
        ap = 0.0
        for t in torch.linspace(0,1,11):
            p = precisions[recalls >= t].max().item() if (recalls >= t).any() else 0
            ap += p
        ap /= 11

        return precisions[-1].item(), recalls[-1].item(), ap
    else:
        print(f"Attack success rate:{len(succquerylist)/(len(querylist))}")

        print(f"avg. query:{np.mean(querylist)}")
        print(f"median. query:{np.median(querylist)}")
        print(f"avg. precision:{np.mean(precisionlist)}")
        print(f"median. precision:{np.median(precisionlist)}")
        print(f"avg. recall:{np.mean(recalllist)}")
        print(f"median. recall:{np.median(recalllist)}")
        print(f"avg. ap:{np.mean(aplist)}")
        print(f"median. ap:{np.median(aplist)}")

        print(f"succ avg. query:{np.mean(succquerylist)}")
        print(f"succ median. query:{np.median(succquerylist)}")
        print(f"succ avg. precision:{np.mean(succprecisionlist)}")
        print(f"succ median. precision:{np.median(succprecisionlist)}")
        print(f"succ avg. recall:{np.mean(succrecalllist)}")
        print(f"succ median. recall:{np.median(succrecalllist)}")
        print(f"succ avg. ap:{np.mean(succaplist)}")
        print(f"succ median. ap:{np.median(succaplist)}")

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    paradict={}
    paradict["iou_thresh"] = 0.5
    paradict["batch_size"] = 1 # 2 for clean evaluation, 1 for adv evaluation
    paradict["attack_method"] = "Oursdy"#ADBA or Ours,Oursdy
    paradict["epsilon"] = 0.05 
    paradict["apthresh"] = 0.2
    paradict["budget"] = 1000
    print(f"paradict:{paradict}")
    parser = argparse.ArgumentParser(description='objectDetectionManual')

    parser.add_argument('--zerosign', default=1, type=int,
                        help='replace zero as 1,-1,and 0 for random chosen from [1,-1]')

    parser.add_argument('--replace', default=0, type=int,
                        help='Dataset')
    parser.add_argument('--useadba', default=1, type=int,
                        help='Dataset')
    parser.add_argument('--binaryAnalyze', default=0, type=int,
                        help='Dataset')
    parser.add_argument('--onlyone', default=0, type=int,
                        help='0::both,1:only std sample;2:only low color square;.')
    parser.add_argument('--ablation', default=0, type=int,
                        help='0::None,1:adbasearch.')
    parser.add_argument('--lowtype', default="rcolor", type=str,
                        help='dct,bar,rcolor,std') 
    parser.add_argument('--dwtlevel', default=3, type=float,
                        help='0:no;1;2')  
    parser.add_argument('--dctTrunc', default=4, type=int,
                        help='0:no;1;2')  
    parser.add_argument('--lowfilter', default=0, type=int,
                        help='0:dwt;1:bdct')  
    parser.add_argument('--earlyexit', default=0, type=int,
                        help='Dataset')


    parser.add_argument('--freqratio', default=4, type=int,
                        help='freqratio:1-64')  
    parser.add_argument('--color', default=0, type=int,
                        help='freqratio:1-64')  

    parser.add_argument('--blocksize', default=8, type=int,
                        help='')   
    parser.add_argument('--deviceid', default="3", type=str,
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
    parser.add_argument('--early', default='1', type=str,
                        help='early stopping (stop attack once the adversarial example is found)')

    parser.add_argument('--defense', default=0, type=int,
                        help='')
    parser.add_argument('--stepp', default=1, type=float,
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
    args = parser.parse_args()
    args.budget = paradict["budget"]
    print(f"args:{args}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval().cuda()

    # 数据集 (示例COCO val2017)

    dataset = CocoDetection(
        root="/data/COCO/val2017/",
        annFile="/data/COCO/annotations/instances_val2017.json",
        transform=T.ToTensor()
    )
    dataloader = DataLoader(dataset, batch_size=paradict["batch_size"], shuffle=False,
                            collate_fn=lambda x: tuple(zip(*x)))

    # 评估
    
    precision, recall, mAP = evaluate_dataset(model, dataloader, device,paradict = paradict,args=args)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, mAP@0.5: {mAP:.3f}")
