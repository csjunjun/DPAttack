import os

import torch,math
import numpy as np
import sys,argparse
from torchvision.utils import save_image

from ..tools.utils import setSeed,get_tracker,get_pert_amp
from ..tools.fetchmodel import load_adv_imagenet




parser = argparse.ArgumentParser(description='attack imagenet_CGBAimgs')
parser.add_argument('--attack_method', default='BFS', type=str,
                    help='ttba,rays,hsja,bounce,surfree,tangent,fgsmgrad,BFS')
parser.add_argument('--targetn', default='Resnet50', type=str,
                    help='Resnet50,vit,inv3,wrs50,Densenet121,SwinV2T,efficient,ConvNextBase')
parser.add_argument('--deviceid', default='0', type=str,
                    help='')
parser.add_argument('--targetimage', default=0, type=int,
                    help='')
parser.add_argument('--constraint', default='np.inf', type=str,
                    help='np.inf,2')
parser.add_argument('--defense', default=0, type=float,
                    help='attack strength') 

parser.add_argument('--maxnumevals', default=50, type=float,
                    help='attack strength') 
parser.add_argument('--maxquery', default=1000, type=float,
                    help='attack strength')

parser.add_argument('--threshold', default=5, type=float,
                    help='attack strength') 
parser.add_argument('--binaryAnalyze', default=0, type=float,
                    help='attack strength') 

parser.add_argument('--imgnum', default=200, type=float,
                    help='attack strength')
parser.add_argument('--seed', default=0, type=float,
                    help='attack strength')
parser.add_argument('--raysFreq', action='store_true')
parser.add_argument('--raysfilter', action='store_true')
parser.add_argument('--savep', action='store_true')

args = parser.parse_args()
if args.attack_method=="ttba":
    parser_2 = argparse.ArgumentParser(description='Hard Label Attacks')
    #parser.add_argument('--dataset', default='mnist', type=str, help='Dataset')
    parser_2.add_argument('--dataset', default='resnet50', type=str, help='Dataset')
    parser_2.add_argument('--targeted', default=0, type=int, help='targeted-1 or untargeted-0')
    parser_2.add_argument('--norm', default='k', type=str, help='Norm for attack, k or l2')
    parser_2.add_argument('--epsilon', default=0.3, type=float, help='attack strength')
    parser_2.add_argument('--budget', default=10000, type=int, help='Maximum query for the attack norm k')
    parser_2.add_argument('--early', default=1, type=int, help='early stopping (stop attack once the adversarial example is found)')
    parser_2.add_argument('--remember', default=0, type=int, help='if remember adversarial examples.')
    parser_2.add_argument('--imgnum', default=200, type=int, help='Number of samples to be attacked from test dataset.')
    parser_2.add_argument('--beginIMG', default=234, type=int, help='begin test img number')
    parser_2.add_argument('--RGB', default=['RGB'], type=str, help='List of RGB channels (e.g., RG)')
    parser_2.add_argument('--binaryM', default=1, type=int, help='binary search mod, mid 0 or median 1.')
    parser_2.add_argument('--initDir', default=1, type=int, help='initial direction, 1,-1,and 0 for random, 2 for 1-11-1...')

    args_2 = parser_2.parse_args()
    args_2.epsilon = args.threshold
    args_2.budget = args.maxquery
    args_2.imgnum = args.imgnum
    print(args_2)

os.environ["CUDA_VISIBLE_DEVICES"]=args.deviceid
print(args)
setSeed(args.seed)
attack_method = args.attack_method
targetn =args.targetn
raysFreq=args.raysFreq
raysfilter=args.raysfilter
maxquery=args.maxquery
threshold = args.threshold
savep = args.savep
imgnum=args.imgnum
max_num_evals = args.maxnumevals


if savep:
    outputdir=f"outputadv_{targetn}_{attack_method}_maxq{maxquery}_logstd_freqSearch"
    print(f"output to {outputdir}")
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)



print(paradict)
net = load_adv_imagenet([args.targetn])[0]

linf_con = args.threshold
target_label =-1

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

if args.attack_method=="ttba":
    from TtBA import Attacker as TTBA
elif args.attack_method=="BFS":
    from  ..tools.jpegdct import DiffJPEG,block_splitting
    from ..models.OursClass import getzigzagcor,getZigzagMeanStd
    diffj = DiffJPEG(bs=8)
    blocksize = 8
    bsp = block_splitting(k=blocksize)
    target_model = args.targetn.lower()
    freqmask_list = []
    savepath = "../data/{target_model}_freqperturbresult_0.05_5" #
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    for i in range(0,blocksize**2):
        m,n= getzigzagcor(i,rows=blocksize,columns=blocksize)
        freqmask= torch.zeros((1,(224//blocksize)**2,blocksize,blocksize))
        freqmask[:,:,m,n]=1
        freqmask_list.append(freqmask)
elif args.attack_method == 'rays':
    from rays import RayS 
    if args.binaryAnalyze==1:
        analyzeGrad = True
    else:
        analyzeGrad=False
    rayssavep=f"../data/GradSignSimilarity/FGSMGrad/hrays_ce_{args.targetn}" 
    if not os.path.exists(rayssavep):
        os.mkdir(rayssavep)
        print(f"create dir {rayssavep}")
    attack_algo = RayS(net, epsilon=threshold,order=args.constraint,freqSearch=raysFreq,filterhis=raysfilter,analyzeGrad=analyzeGrad,netname=args.targetn,binaryAnalyze=args.binaryAnalyze)
elif args.attack_method == 'hsja':
    from hsja import hsja

elif args.attack_method == "bounce":
    from bounceAttack import bounce
elif args.attack_method == "surfree":
    from surfree.surfree import SurFree
    attack_algo = SurFree(T=3,theta_max=30,n_ortho=100,steps=paradict["maxquery"],quantification=False, max_queries=paradict["maxquery"])
elif args.attack_method == "tangent":
    sys.path.append("baselines/TangentAttack/tangent_attack_semiellipsoid")
    from attack import EllipsoidTangentAttack 

    if args.constraint=='l2':
        gamma = 1000
    elif args.constraint=='linf':
        gamma=10000
    attack_algo = EllipsoidTangentAttack(net,'ImageNet',0, 1.0,224,224,3,args.constraint,args.threshold,1.1,paradict["maxquery"],gamma=1000,stepsize_search='geometric_progression',max_num_evals=max_num_evals,init_num_evals=100,maximum_queries=paradict["maxquery"],verify_tangent_point=False)
query_list,l2_list,linf_list,succ_list,all_list = [],[],[],[],[]
l2list_all=[]
iTotal = 0
iteridx = 0
from PIL import Image 
from torchvision import transforms
trans = transforms.Compose([
    transforms.Resize((224,224)),
        transforms.ToTensor()
        ])
def mysortkey(filename:str):
    return int(filename.split("_")[2].split(".")[0])  

imgbase = '../data/imagenet/val'
imagelist = [f for f in os.listdir(imgbase)]
imagelist.sort(key=mysortkey)
if args.binaryAnalyze == 1 or args.attack_method== "BFS" or args.attack_method== "fgsmgrad":
    imagelist=imagelist[2000:]
    args.imgnum=1000
    imgnum = 1000
    saveeeep=f"../data/targetinitAE_{args.targetn}"
    if not os.path.exists(saveeeep):
        os.mkdir(saveeeep)
imagelist_final2000 = imagelist[-2000:]
ground_truth  = open(os.path.join('../data/imagenet/imagenet_test.txt'), 'r').read().split('\n')
if args.binaryAnalyze == 1 or args.attack_method== "BFS" or args.attack_method== "fgsmgrad":

    ttt = 2000
else:
    ttt = 0

fail,recog_succ=0,0
blacklight_succ_detection,blacklight_fail_detection = 0,0
blacklight_first_detects=[]
mselos = torch.nn.MSELoss()
celos = torch.nn.CrossEntropyLoss()
zeronum_list=[]
pixelnum = 3*224*224
celoss = torch.nn.CrossEntropyLoss()
cossimlist=[]
initqlist=[]
advsucc = 0 
advfail=0
for filename in imagelist:
    blacklight_count = 0
    imgpath = "{}/{}".format(imgbase,filename)
    ground_name_label = ground_truth[ttt] 
    ttt +=1  
    ground_label_split_all =  ground_name_label.split
    
    ground_label_split =  ground_name_label.split()
    
    ground_label =  ground_name_label.split()[1]
    ground_name =  ground_name_label.split()[0]
    innerl = 0
    while not filename==ground_name:
        ground_name_label = ground_truth[ttt]            
        ground_label_split_all =  ground_name_label.split
        
        ground_label_split =  ground_name_label.split()
        
        ground_label =  ground_name_label.split()[1]
        ground_name =  ground_name_label.split()[0]
        innerl +=1
        if innerl>1000:
            break
        ttt+=1
    label = torch.tensor([int(ground_label)])
    test_secrets_pil = Image.open(imgpath)
    test_secrets_pil = test_secrets_pil.convert("RGB")
    batch = trans(test_secrets_pil).unsqueeze(0)
 
    batch = batch.cuda()
    label = label.cuda()
    with torch.no_grad():
        output = net(batch)
    pre=torch.argmax(output,dim=1)
    clean_gt_score=output[0][label]
    if target_label>=0 and pre==target_label:
        fail += 1
        continue
    elif pre != label:
        fail += 1
        continue
    recog_succ +=1 
    if args.defense == 1:
        
        tracker = get_tracker(batch, window_size= 50, hash_kept= 50, roundto= 50, step_size= 25, workers= 5)
    else:
        tracker = None
    print(f"imgpath:{imgpath}")    

    if args.attack_method=="ttba":
        attack_algo = TTBA(args_2,model=net,  attack_method='TtBA',dim_reduc_factor=4,
                    iteration=args.maxquery, initial_query=30, tol=1e-4, sigma=3e-4,measure=args.constraint)

        res = attack_algo.attack( batch,label,iTotal,tar_img=None, tar_label=None,tracker=tracker)
        blacklight_count = attack_algo.blacklight_count
        blacklight_first_detect = attack_algo.blacklight_first_detect
        
        succ = True if attack_algo.success==1 else False
        query = attack_algo.queries
        if blacklight_count==0:
            adv = attack_algo.Img_result[2]
        else:
            adv = None
            

    elif args.attack_method=="BFS":
        with torch.no_grad():
            yc,cb,cr = diffj(batch.detach().cpu(),forged=False,batch=True)
        ycbcr = [yc,cb,cr]
        name=["y","cb","cr"]
        i=0
        for it in ycbcr:
            it = it.cuda()
            maxamp = torch.max(it)
            mu,std,_ = getZigzagMeanStd(it[0])
               
            print(f"channel {i},maxamp:{maxamp}")
            pre_list,pre_l2_list,linflosslist,l2losslist = [],[],[],[]
            freqi = 0
            for freqmask in freqmask_list:
                maxamp = 10000000000000
                freqi += 1
                perturb_amp = get_pert_amp(it,maxamp,freqmask,type="dct")
                if i==0:
                    recon_img = diffj.rec(perturb_amp.cpu(),cb,cr,224,224) 
                elif i==1:
                    recon_img = diffj.rec(yc,perturb_amp.cpu(),cr,224,224) 
                elif i==2:
                    recon_img = diffj.rec(yc,cb,perturb_amp.cpu(),224,224) 
                recon_img=recon_img.cuda()
                perturb_pixel = recon_img-batch
                perturb_pixel = torch.clamp(perturb_pixel,-0.05,0.05)
                perturb_pixel_unit = perturb_pixel/torch.norm(perturb_pixel)
                adv = batch + perturb_pixel
               # adv_l2 = batch + perturb_pixel_unit*30
                adv_l2 = batch + perturb_pixel_unit*5
                adv = torch.clamp(adv, 0, 1)
                adv_l2 = torch.clamp(adv_l2, 0, 1)
                with torch.no_grad():
                    output = net(adv)
                    output_l2 = net(adv_l2)
                pre = torch.argmax(output, dim=1)
                pre_l2 = torch.argmax(output_l2, dim=1)
                pre_list.append(float(pre))
                pre_l2_list.append(float(pre_l2))
                linfloss = celoss(output,label)
                l2loss = celoss(output_l2,label)
                linflosslist.append(float(linfloss))
                l2losslist.append(float(l2loss))

            np.save(f"{savepath}/{filename.split('.')[0]}_{int(label)}_pre_list_{name[i]}_randn.npy",pre_list) 
            np.save(f"{savepath}/{filename.split('.')[0]}_{int(label)}_pre_l2_list_{name[i]}_randn.npy",pre_l2_list) 
            np.save(f"{savepath}/{filename.split('.')[0]}_{int(label)}_linfceloss_list_{name[i]}_randn.npy",linflosslist) 
            np.save(f"{savepath}/{filename.split('.')[0]}_{int(label)}_l2celoss_list_{name[i]}_randn.npy",l2losslist) 

            i+=1

        iTotal+=1
        if iTotal>=imgnum:
            break        
        continue

    elif args.attack_method=="fgsmgrad":
        images = batch.clone().detach().cuda()
        labels = label.clone().detach().cuda()
        inneri = iTotal
        images.requires_grad = True
        outputs = net(images)
        
        # Calculate loss
        cost = celos(outputs, labels)

        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        if not os.path.exists(f"../data/GradSignSimilarity/FGSMGrad/cleanimgCEloss_{args.targetn}"):
            os.mkdir(f"../data/GradSignSimilarity/FGSMGrad/cleanimgCEloss_{args.targetn}")
        if not os.path.exists(f"../data/GradSignSimilarity/FGSMGrad/cleanimgCEloss_{args.targetn}/{filename.split('.')[0]}.pth"):
            torch.save(grad.detach().cpu().numpy(),f"../data/GradSignSimilarity/FGSMGrad/cleanimgCEloss_{args.targetn}/{filename.split('.')[0]}.pth")
        iTotal+=1
        if iTotal>=imgnum:
            break

        continue


    elif args.attack_method == 'rays':
    
        if analyzeGrad==True:
            adv, succ,query ,blacklight_count,blacklight_first_detect,cossimlist= attack_algo.attack_hard_label(batch,label,\
                            target=target_label,query_limit = paradict['maxquery'],filename=filename.split(".")[0],tracker=tracker)
        else:
            if args.binaryAnalyze==2:
                adv, succ,query ,blacklight_count,blacklight_first_detect,initq= attack_algo.attack_hard_label(batch,label,\
                            target=target_label,query_limit = paradict['maxquery'],filename=filename.split(".")[0],tracker=tracker)
                initqlist.append(initq)
            else:
                adv, succ,query ,blacklight_count,blacklight_first_detect= attack_algo.attack_hard_label(batch,label,\
                            target=target_label,query_limit = paradict['maxquery'],filename=filename.split(".")[0],tracker=tracker)

    elif args.attack_method == "surfree":
        epsilons=[args.threshold]
        adv, succ,query = attack_algo(net, batch, label.unsqueeze(0), epsilons=epsilons,constraint=args.threshold)
    elif args.attack_method == 'tangent':
        random_target = np.random.randint(0, 1000)
        random_targetimg = imagelist_final2000[random_target]
        fname_target,random_targetimg_label = ground_truth[2000+random_target].split()
        while int(fname_target.split(".")[0].split("_")[-1])>int(random_targetimg.split(".")[0].split("_")[-1]):
            random_target -= 1
            fname_target,random_targetimg_label = ground_truth[2000+random_target].split()
        while int(fname_target.split(".")[0].split("_")[-1])<int(random_targetimg.split(".")[0].split("_")[-1]):
            random_target += 1
            fname_target,random_targetimg_label = ground_truth[2000+random_target].split()
        assert fname_target==random_targetimg
        random_targetimg_label = int(random_targetimg_label)
        random_targetimg_tensor = "{}/{}".format(imgbase,random_targetimg)
        random_targetimg_tensor = Image.open(random_targetimg_tensor)
        random_targetimg_tensor = random_targetimg_tensor.convert("RGB")
        random_targetimg_tensor = trans(random_targetimg_tensor).unsqueeze(0)
        
        adv, query, _, _, _,succ = attack_algo.attack( 0, batch.cpu(), random_targetimg_tensor, label.cpu(), None)
        if adv is not None:
            adv = adv.unsqueeze(0).cuda()
    elif args.attack_method == 'bounce':

        if args.targetimage==1:
            random_target = np.random.randint(0, 1000)
            random_targetimg = imagelist_final2000[random_target]
            fname_target,random_targetimg_label = ground_truth[2000+random_target].split()
            while int(fname_target.split(".")[0].split("_")[-1])>int(random_targetimg.split(".")[0].split("_")[-1]):
                random_target -= 1
                fname_target,random_targetimg_label = ground_truth[2000+random_target].split()
            while int(fname_target.split(".")[0].split("_")[-1])<int(random_targetimg.split(".")[0].split("_")[-1]):
                random_target += 1
                fname_target,random_targetimg_label = ground_truth[2000+random_target].split()
            assert fname_target==random_targetimg
            random_targetimg_label = int(random_targetimg_label)
            random_targetimg_tensor = "{}/{}".format(imgbase,random_targetimg)
            random_targetimg_tensor = Image.open(random_targetimg_tensor)
            random_targetimg_tensor = random_targetimg_tensor.convert("RGB")
            random_targetimg_tensor = trans(random_targetimg_tensor).unsqueeze(0)
            adv,succ,query,blacklight_count,blacklight_first_detect= bounce(net, args.threshold,batch[0].detach().cpu().numpy(), constraint = args.constraint,num_iterations = paradict["maxquery"],max_num_evals=max_num_evals,target_image=random_targetimg_tensor[0].detach().cpu().numpy(),gamma=10,tracker=tracker)        # succ = True
        else:
            adv,succ,query ,blacklight_count,blacklight_first_detect= bounce(net, args.threshold,batch[0].detach().cpu().numpy(), constraint = args.constraint,num_iterations = paradict["maxquery"],max_num_evals=max_num_evals,target_image=None,gamma=10,tracker=tracker) 
        if adv is not None:
            adv = torch.tensor(adv).float().cuda()



    elif args.attack_method == 'hsja':
        if args.targetimage==1:
            random_target = np.random.randint(0, 1000)
            random_targetimg = imagelist_final2000[random_target]
            fname_target,random_targetimg_label = ground_truth[2000+random_target].split()
            while int(fname_target.split(".")[0].split("_")[-1])>int(random_targetimg.split(".")[0].split("_")[-1]):
                random_target -= 1
                fname_target,random_targetimg_label = ground_truth[2000+random_target].split()
            while int(fname_target.split(".")[0].split("_")[-1])<int(random_targetimg.split(".")[0].split("_")[-1]):
                random_target += 1
                fname_target,random_targetimg_label = ground_truth[2000+random_target].split()
            assert fname_target==random_targetimg
            random_targetimg_label = int(random_targetimg_label)
            random_targetimg_tensor = "{}/{}".format(imgbase,random_targetimg)
            random_targetimg_tensor = Image.open(random_targetimg_tensor)
            random_targetimg_tensor = random_targetimg_tensor.convert("RGB")
            random_targetimg_tensor = trans(random_targetimg_tensor).unsqueeze(0)
            istest = True if args.binaryAnalyze==1 else False
            adv,succ,query,blacklight_count,blacklight_first_detect = hsja(net, args.threshold,batch[0].detach().cpu().numpy(), constraint =args.constraint,num_iterations = paradict["maxquery"],max_num_evals=max_num_evals,target_image=random_targetimg_tensor,test=istest,tracker=tracker)             # succ = True
        else:
            adv,succ,query,blacklight_count,blacklight_first_detect = hsja(net, args.threshold,batch[0].detach().cpu().numpy(), constraint =args.constraint,num_iterations = paradict["maxquery"],max_num_evals=max_num_evals,tracker=tracker)             # succ = True
        if adv is not None:
            adv = torch.tensor(adv).float().cuda()
        
    else:

        with torch.no_grad():
            adv,succ,query = attack_algo.forward(batch, label, target_label,imgidx=iteridx)
    


    if succ and query<=paradict["maxquery"]:
        advl2 = torch.norm(adv-batch)
        advlinf = torch.norm(adv-batch,p=float('inf'))
        if args.constraint=="linf":
            dsdf = advlinf
        else:
            dsdf = advl2
        if  torch.round(dsdf,decimals=2)<=args.threshold:
                
            
            succ_list.append(iteridx)
            query_list.append(int(query))
            l2_list.append(float(advl2))
            linf_list.append(float(advlinf))
            all_list.append(int(query))
            if savep:
                save_image(adv,f'../data/{outputdir}/{filename.split(".")[0]}.png')
            print('Img:{} {} succ:{} query:{} advL2:{:.6f} advLinf:{:.6f} '.format(str(iTotal+1),imgpath,succ,query, torch.mean(advl2).data,torch.mean(advlinf).data))
        else:
            succ = False
            
            print('Img:{} f{} succ:{} query:{} advL2:{:.6f} advLinf:{:.6f} '.format(str(iTotal+1),imgpath,succ,query, torch.mean(advl2).data,torch.mean(advlinf).data))
        l2list_all.append(float(advl2))

    if not succ:
        if adv is not None:
            advl2 = torch.norm(adv-batch)
            l2list_all.append(float(advl2))
            print(f"image:{iTotal+1},{imgpath} failed query:{query} advL2:{advl2}")
        all_list.append(paradict["maxquery"])
        
    iTotal+=1 
    iteridx+=1
    asr = len(succ_list)/iTotal
    if len(query_list)>0:
        avg_q = np.mean(np.asarray(query_list))
        median_q = np.median(np.asarray(query_list))

        avg_l2 = np.mean(np.asarray(l2_list))
        avg_linf = np.mean(np.asarray(linf_list))

        median_l2 = np.median(np.asarray(l2_list))
        median_linf = np.median(np.asarray(linf_list))

        all_avg_l2 = np.mean(np.asarray(l2list_all))
        all_median_l2 = np.median(np.asarray(l2list_all))
    if len(query_list)>0:
        print("Succ rate:{}/{}={:.4f}".format(len(succ_list),iTotal,asr))
        print("Succ Avg.query:{:.4f}".format(avg_q))
        print("Succ Avg.l2_list:{:.4f}".format(avg_l2))
        print("Succ Avg.linf_list:{:.4f}".format(avg_linf))    

        print("Succ Median.query:{:.4f}".format(median_q))
        print("Succ Median.l2_list:{:.4f}".format(median_l2))
        print("Succ Median.linf_list:{:.4f}".format(median_linf))

            
        print("All Avg.query:{:.4f}".format(np.mean(np.asarray(all_list)))) 
        print("All Median.query:{:.4f}".format(np.median(np.asarray(all_list))))  

        print("All Avg.l2:{:.4f}".format(all_avg_l2)) 
        print("All Median.l2:{:.4f}".format(all_median_l2))       
        print(f"Recg asr:{recog_succ/(recog_succ+fail)}")

    if blacklight_count is not None and blacklight_count>0:
        blacklight_succ_detection += 1
        blacklight_first_detects.append(blacklight_first_detect)
    else:
        blacklight_fail_detection += 1
    print(f"blacklight_succ_detection={blacklight_succ_detection},blacklight_fail_detection={blacklight_fail_detection}")
    print(f"blacklight_detect_ratios={blacklight_succ_detection/(blacklight_succ_detection+blacklight_fail_detection)}")
    print(f"avg blacklight_first_detects={np.mean(np.array(blacklight_first_detects))}")
    print(f"median blacklight_first_detects={np.median(np.array(blacklight_first_detects))}")

    if len(cossimlist)>0 and analyzeGrad == True:
        np.save(f"{rayssavep}/{filename.split('.')[0]}.npy",np.array(cossimlist))

    if iTotal>=imgnum:
        break

print("=======================================================================================")
print("Succ rate:{:.4f}".format(asr))
print("Succ Avg.query:{:.4f}".format(avg_q))
print("Succ Avg.l2_list:{:.4f}".format(avg_l2))
print("Succ Avg.linf_list:{:.4f}".format(avg_linf))    

print("Succ Median .query:{:.4f}".format(median_q))
print("Succ Median.l2_list:{:.4f}".format(median_l2))
print("Succ Median.linf_list:{:.4f}".format(median_linf))   

print("All Avg.l2:{:.4f}".format(all_avg_l2)) 
print("All Median.l2:{:.4f}".format(all_median_l2))  

print("All Avg.query:{:.4f}".format(np.mean(np.asarray(all_list)))) 
print("All Median.query:{:.4f}".format(np.median(np.asarray(all_list)))) 
print(f"Recg asr:{recog_succ/(recog_succ+fail)}")
print(f"blacklight_succ_detection={blacklight_succ_detection},blacklight_fail_detection={blacklight_fail_detection}")
print(f"blacklight_detect_ratios={blacklight_succ_detection/(blacklight_succ_detection+blacklight_fail_detection)}")
print(f"avg blacklight_first_detects={np.mean(np.array(blacklight_first_detects))}")
print(f"median blacklight_first_detects={np.median(np.array(blacklight_first_detects))}")

