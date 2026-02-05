import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import numpy as np
import sys 
from torchvision import transforms
from ..tools.fetchmodel import load_adv_imagenet
from ..models.clipmodel import MyCliptModel
from ..tools.utils import setSeed
torch.set_num_threads(1)
seed = 0
setSeed(seed)
attack_method = 'hsja'#rays,hsja,bounce
targetimg=0 #for bounce,

paradict = {
    'linf_con':0.05,
    'maxquery':1000,
    'target_label':-1,
    'attack_method':attack_method, 
    
    
}
print(paradict)
device = "cuda" if torch.cuda.is_available() else "cpu"   
clipM = 'ViT-H-14-CLIPA-336'#
clipdataset='datacomp1b' #


import open_clip
net, _, preprocess = open_clip.create_model_and_transforms(clipM, \
                                                                pretrained=clipdataset,device=device)
net = MyCliptModel(net,clipM,clipdataset,device)



linf_con = paradict['linf_con']
target_label = paradict['target_label']
paradict['text_features']=net.text_features

paradict['transform_clip'] = net.transform_clip

if  paradict['attack_method'] =='ttba':
    import argparse
    from TtBA import Attacker as TTBA
    parser_2 = argparse.ArgumentParser(description='Hard Label Attacks')
    parser_2.add_argument('--dataset', default='mnist', type=str, help='Dataset')
    parser_2.add_argument('--targeted', default=0, type=int, help='targeted-1 or untargeted-0')
    parser_2.add_argument('--norm', default='k', type=str, help='Norm for attack, k or l2')
    parser_2.add_argument('--epsilon', default=0.05, type=float, help='attack strength')
    parser_2.add_argument('--budget', default=40000, type=int, help='Maximum query for the attack norm k')
    parser_2.add_argument('--early', default=1, type=int, help='early stopping (stop attack once the adversarial example is found)')
    parser_2.add_argument('--remember', default=0, type=int, help='if remember adversarial examples.')
    parser_2.add_argument('--imgnum', default=200, type=int, help='Number of samples to be attacked from test dataset.')
    parser_2.add_argument('--beginIMG', default=0, type=int, help='begin test img number')
    parser_2.add_argument('--RGB', default=['RGB'], type=str, help='List of RGB channels (e.g., RG)')
    parser_2.add_argument('--binaryM', default=1, type=int, help='binary search mod, mid 0 or median 1.')
    parser_2.add_argument('--initDir', default=1, type=int, help='initial direction, 1,-1,and 0 for random, 2 for 1-11-1...')

    args_2 = parser_2.parse_args()
    args_2.epsilon = linf_con
    args_2.budget = paradict['maxquery']
    args_2.imgnum = args_2.imgnum
    print(args_2)

elif paradict['attack_method'] == 'rays':
    from attacks.rays import RayS 
    attack_algo = RayS(net, epsilon=linf_con,order=np.inf,freqSearch=False,filterhis=False)
elif paradict['attack_method'] == 'hsja':
    from hsjaCorrect import hsja 
elif paradict['attack_method'] == 'bounce':
    from bounceAttack import bounce 
query_list,l2_list,linf_list,succ_list,all_list = [],[],[],[],[]
iTotal = 0
iteridx = 0
from PIL import Image 
from torchvision import transforms

with open(f"{net.basepp}/cliptest1000_withoutimagenetclasswithidx.txt","r") as f:
    imagelistall = f.read().splitlines()

imagelist = imagelistall[:300]
imagelist_final1000 = imagelistall[-1000:]
trans = transforms.Compose([
transforms.Resize([336,336],transforms.InterpolationMode.BILINEAR),

        transforms.ToTensor()
        ])
for imgp in imagelist:
    imgpath,labelname,labelinallwithimagenet = imgp.split(",")
    #imgpath = imgpath.replace("yourpath/","..")
    label = torch.tensor([int(list(net.classlist).index(labelname))])
    
    test_secrets = Image.open(imgpath)
    test_secrets = test_secrets.convert("RGB")
    batch = trans(test_secrets).unsqueeze(0)
 
    batch = batch.cuda()
    label = label.cuda()
    if label==target_label and target_label!=-1:
        continue
    if target_label == -1:
        predprob = net(batch)
        pred = torch.argmax(predprob)
        if pred != label:
            continue
    
    if paradict['attack_method'] == 'rays':
        adv, succ,query = attack_algo.attack_hard_label(batch,label,target=target_label,query_limit = paradict['maxquery'],filename=imgp.split(".")[0])
    elif paradict['attack_method'] == 'ttba':
        attack_algo = TTBA(args_2,model=net,  attack_method='TtBA',dim_reduc_factor=4,
                    iteration=paradict['maxquery'], initial_query=30, tol=1e-4, sigma=3e-4,measure='linf')

        res = attack_algo.attack( batch,label,iTotal,tar_img=None, tar_label=None)
        succ = True if attack_algo.success==1 else False
        query = attack_algo.queries
        adv = attack_algo.Img_result[2]
            

    elif paradict['attack_method'] == 'hsja':
        if paradict['maxquery']<=100:
            max_num_evals = 10 
        elif paradict['maxquery']<=500:
            max_num_evals = 50
        else:
            max_num_evals = 100
        adv,succ,query,_,_ = hsja(net, paradict['linf_con'],batch[0].detach().cpu().numpy(), constraint = 'linf',num_iterations = paradict["maxquery"],max_num_evals=max_num_evals)        # succ = True
        if adv is not None:
            adv = torch.tensor(adv).float().cuda()
    elif paradict['attack_method'] == 'bounce':
        if paradict['maxquery']<=100:
            max_num_evals = 10 
        elif paradict['maxquery']<=500:
            max_num_evals = 50
        else:
            max_num_evals = 100
        if targetimg==1:
            random_target = np.random.randint(0, 1000)
            random_targetimg = imagelist_final1000[random_target]
            timgpath,tlabelname,tlabelinallwithimagenet = random_targetimg.split(",")
            #timgpath = timgpath.replace("yourpath/","..")

            tlabel = torch.tensor([int(list(net.classlist).index(tlabelname))])
            cou=0
            while float(tlabel)==float(label):
                random_target = np.random.randint(0, 1000)
                random_targetimg = imagelist_final1000[random_target]
                timgpath,tlabelname,tlabelinallwithimagenet = random_targetimg.split(",")
                #timgpath = timgpath.replace("yourpath/","..")

                tlabel = torch.tensor([int(list(net.classlist).index(tlabelname))])
                cou+=1
                if cou==1000:
                    print("No random target image found!")
                    break
            random_targetimg_tensor = Image.open(timgpath)
            random_targetimg_tensor = random_targetimg_tensor.convert("RGB")

            random_targetimg_tensor = trans(random_targetimg_tensor).unsqueeze(0)

            random_targetimg_tensor = random_targetimg_tensor.cuda()
            adv,succ,query,_,_ = bounce(net, paradict['linf_con'],batch[0].detach().cpu().numpy(), constraint = 'linf',num_iterations = paradict["maxquery"],max_num_evals=max_num_evals,target_image=random_targetimg_tensor[0].detach().cpu().numpy(),gamma=10)        # succ = True
        else:
            adv,succ,query,_,_ = bounce(net, paradict['linf_con'],batch[0].detach().cpu().numpy(), constraint = 'linf',num_iterations = paradict["maxquery"],max_num_evals=max_num_evals,target_image=None,gamma=10) 
        if adv is not None:
            adv = torch.tensor(adv).float().cuda()

    else:

        with torch.no_grad():
            adv,succ,query = attack_algo.forward(batch, label, target_label,imgidx=iteridx)



    if succ and query<=paradict['maxquery'] and adv is not None:
        advl2 = torch.norm(adv-batch)
        advlinf = torch.norm(adv-batch,p=float('inf'))

        succ_list.append(iteridx)
        query_list.append(int(query))
        l2_list.append(float(advl2))
        linf_list.append(float(advlinf))
        all_list.append(int(query))
        print('Img:{} succ:{} query:{} advL2:{:.6f} advLinf:{:.6f} '.format(str(iTotal+1),succ,query, torch.mean(advl2).data,torch.mean(advlinf).data))

    else:
        all_list.append(paradict['maxquery'])
        print('Img:{} succ:{} query:{}  '.format(str(iTotal+1),succ,query))
    
    
    #from torchvision.utils import save_image
    iTotal+=1 
    iteridx+=1
    
    asr = len(succ_list)/iTotal
    avg_q = np.mean(np.asarray(query_list))
    median_q = np.median(np.asarray(query_list))

    avg_l2 = np.mean(np.asarray(l2_list))
    avg_linf = np.mean(np.asarray(linf_list))

    median_l2 = np.median(np.asarray(l2_list))
    median_linf = np.median(np.asarray(linf_list))
    if iTotal%10==0:
        print("Succ rate:{:.4f}".format(asr))
        print("Succ Avg.query:{:.4f}".format(avg_q))
        print("Succ Avg.l2_list:{:.4f}".format(avg_l2))
        print("Succ Avg.linf_list:{:.4f}".format(avg_linf))    

        print("Succ Median.query:{:.4f}".format(median_q))
        print("Succ Median.l2_list:{:.4f}".format(median_l2))
        print("Succ Median.linf_list:{:.4f}".format(median_linf))

          
        print("All Avg.query:{:.4f}".format(np.mean(np.asarray(all_list)))) 
        print("All Median.query:{:.4f}".format(np.median(np.asarray(all_list))))       
    if iTotal>=200:
        break
print("=======================================================================================")
print("Succ rate:{:.4f}".format(asr))
print("Succ Avg.query:{:.4f}".format(avg_q))
print("Succ Avg.l2_list:{:.4f}".format(avg_l2))
print("Succ Avg.linf_list:{:.4f}".format(avg_linf))    

print("Succ Median .query:{:.4f}".format(median_q))
print("Succ Median.l2_list:{:.4f}".format(median_l2))
print("Succ Median.linf_list:{:.4f}".format(median_linf))   
 
print("All Avg.query:{:.4f}".format(np.mean(np.asarray(all_list)))) 
print("All Median.query:{:.4f}".format(np.median(np.asarray(all_list)))) 