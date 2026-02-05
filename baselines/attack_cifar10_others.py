import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch
import numpy as np
import sys 
from ..tools.utils import setSeed
from ..tools.fetchmodel import featchCifar10Models
seed = 0
setSeed(seed)

paradict = {
    'targetimage':0, #for bounce
    'constraint':'linf',
    'linf_con':0.05,
    'targetn':'res18',
    'target_label':-1, #-1 for untargeted, torch.tensor([0]) or other labels for targeted
    'attack_method':'ttba', #tangent,rays,nrays,bounce,hsja,ttba
    'maxquery':100,
    'nes_samples':2,
    'sample_per_draw':30, #default
    'step_size':1/255,#default   
}
print(paradict)
net = featchCifar10Models(paradict['targetn'])
linf_con = paradict['linf_con']
target_label = paradict['target_label']

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

if paradict['attack_method'] =='rays' or paradict['attack_method'] =='nrays':
    from rays import RayS 
    attack_algo = RayS(net, epsilon=paradict['linf_con'],freqSearch=False,filterhis=False)
elif paradict['attack_method'] =='bounce':
    from bounceAttack import bounce
elif paradict['attack_method'] == 'tangent':
    #you have to git clone https://github.com/machanic/TangentAttack.git
    sys.path.append("yourpath/TangentAttack/tangent_attack_semiellipsoid")
    from attack import EllipsoidTangentAttack 
    if paradict['maxquery']<=100:
        max_num_evals = 10 
    elif paradict['maxquery']<=500:
        max_num_evals = 50
    else:
        max_num_evals = 100

    if paradict['constraint']=='l2':
        gamma = 1
    elif paradict['constraint']=='linf':
        gamma=100
    attack_algo = EllipsoidTangentAttack(net,'CIFAR-10',0, 1.0,32,32,3,paradict["constraint"],paradict['linf_con'],1.1,paradict["maxquery"],gamma=1000,stepsize_search='geometric_progression',max_num_evals=max_num_evals,init_num_evals=100,maximum_queries=paradict["maxquery"],verify_tangent_point=False)

elif paradict['attack_method'] == 'hsja':
    from hsja import hsja 
elif paradict['attack_method']=='ttba':
    import argparse
    parser_2 = argparse.ArgumentParser(description='Hard Label Attacks')
    parser_2.add_argument('--dataset', default='cifar10', type=str, help='Dataset')
    parser_2.add_argument('--targeted', default=0, type=int, help='targeted-1 or untargeted-0')
    parser_2.add_argument('--norm', default='k', type=str, help='Norm for attack, k or l2')
    parser_2.add_argument('--epsilon', default=linf_con, type=float, help='attack strength')
    parser_2.add_argument('--budget', default=paradict['maxquery'], type=int, help='Maximum query for the attack norm k')
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
    args_2.imgnum = 200
    print(args_2)

    from TtBA import Attacker as TTBA
query_list,l2_list,linf_list,succ_list = [],[],[],[]
iTotal = 0
fail = 0

def mysortkey(filename:str):
    return int(filename.split("_")[1].split(".")[0])  

imgbase = '../data/cifar10/val'
imagelist = [f for f in os.listdir(imgbase)]
imagelist.sort(key=mysortkey)
#for imgpath,labels in testfiles:
imagelistlast500=imagelist[-500:]
ground_truth  = open(os.path.join('../data/cifar10/val.txt'), 'r').read().split('\n')

iteridx=0
ttt=0
succ_q_list = []
all_q_list = []
succ_l2_list = []
all_l2_list = []
succ_linf_list = []
all_linf_list = []
blacklight_detect_ratios = []
blacklight_first_detects=[]
for filename in imagelist:
    imgpath = "{}/{}".format(imgbase,filename)
        
    ground_name_label = ground_truth[ttt]
    ttt+=1 
    
    ground_label_split_all =  ground_name_label.split
    
    ground_label_split =  ground_name_label.split()
    
    ground_label =  ground_name_label.split()[1]
    ground_name =  ground_name_label.split()[0]
    label = torch.tensor(int(ground_label)).unsqueeze(0)
    test_secrets_pil = np.load(imgpath)
    batch = torch.tensor(test_secrets_pil).unsqueeze(0)
    
    batch = batch.cuda()
    label = label.cuda()
    with torch.no_grad():
        output = net(batch)
    pre=torch.argmax(output,dim=1)
    if target_label>=0 and pre==target_label:
        fail+=1
        continue
    elif pre != label:
        fail+=1
        continue
    if paradict['attack_method'] =='subspace' or paradict['attack_method'] =='prgf':
        adv,succ,query = attack_algo.forward(batch, label, target_label,imgidx=iteridx)
    else:
        with torch.no_grad():
            if paradict['attack_method'] =='rays':
                adv, succ,query,_,_ = attack_algo.attack_hard_label(batch,label,target=target_label,query_limit = paradict['maxquery'])
            elif paradict['attack_method'] =='nrays':
                adv, succ,query,_,_ = attack_algo.attack_hard_label_naive(batch,label,target=target_label,query_limit = paradict['maxquery'])

            elif paradict['attack_method']=='ttba':
                attack_algo = TTBA(args_2,model=net,  attack_method='TtBA',dim_reduc_factor=1,
                            iteration=args_2.budget, initial_query=30, tol=1e-4, sigma=3e-4,measure='linf')

                res = attack_algo.attack( batch,label,iTotal,tar_img=None, tar_label=None)
                succ = True if attack_algo.success==1 else False
                query = attack_algo.queries
                adv = attack_algo.Img_result[2]            
            elif paradict['attack_method'] =='bounce':

                if paradict['maxquery']<=100:
                    #max_num_evals = 10 
                    max_num_evals = 50 
                elif paradict['maxquery']<=500:
                    #max_num_evals = 50
                    max_num_evals = 300
                else:
                    max_num_evals = 100
                    #max_num_evals = 200
                if paradict["targetimage"]==1:
                    random_target = np.random.randint(0, 500)
                    random_targetimg = imagelistlast500[random_target]
                    fname_target,random_targetimg_label = ground_truth[500+random_target].split()
                    while int(fname_target.split(".")[0].split("_")[-1])>int(random_targetimg.split(".")[0].split("_")[-1]):
                        random_target -= 1
                        fname_target,random_targetimg_label = ground_truth[500+random_target].split()
                    while int(fname_target.split(".")[0].split("_")[-1])<int(random_targetimg.split(".")[0].split("_")[-1]):
                        random_target += 1
                        fname_target,random_targetimg_label = ground_truth[500+random_target].split()
                    assert fname_target==random_targetimg
                    random_targetimg_label = int(random_targetimg_label)
                    random_targetimg_tensor = np.load("{}/{}".format(imgbase,random_targetimg))
                    random_targetimg_tensor = torch.tensor(random_targetimg_tensor).unsqueeze(0)

                    adv,succ,query,_,_ = bounce(net, paradict['linf_con'],batch[0].detach().cpu().numpy(), constraint = paradict['constraint'],num_iterations = paradict["maxquery"],max_num_evals=max_num_evals,target_image=random_targetimg_tensor[0].detach().cpu().numpy())        # succ = True
                else:
                    adv,succ,query,_,_ = bounce(net, paradict['linf_con'],batch[0].detach().cpu().numpy(), constraint = paradict['constraint'],num_iterations = paradict["maxquery"],max_num_evals=max_num_evals,target_image=None) 
                if adv is not None:
                    adv = torch.tensor(adv).float().cuda()
            elif paradict['attack_method'] == 'tangent':
                if paradict['maxquery']<=100:
                    max_num_evals = 10 
                elif paradict['maxquery']<=500:
                    max_num_evals = 50
                else:
                    max_num_evals = 100
            
                random_target = np.random.randint(0, 500)
                random_targetimg = imagelistlast500[random_target]
                fname_target,random_targetimg_label = ground_truth[500+random_target].split()
                while int(fname_target.split(".")[0].split("_")[-1])>int(random_targetimg.split(".")[0].split("_")[-1]):
                    random_target -= 1
                    fname_target,random_targetimg_label = ground_truth[500+random_target].split()
                while int(fname_target.split(".")[0].split("_")[-1])<int(random_targetimg.split(".")[0].split("_")[-1]):
                    random_target += 1
                    fname_target,random_targetimg_label = ground_truth[500+random_target].split()
                assert fname_target==random_targetimg
                random_targetimg_label = int(random_targetimg_label)
                random_targetimg_tensor = np.load("{}/{}".format(imgbase,random_targetimg))
                random_targetimg_tensor = torch.tensor(random_targetimg_tensor).unsqueeze(0)

                adv, query, _, _, _,succ = attack_algo.attack( 0, batch.cpu(), random_targetimg_tensor, label.cpu(), None)
                if adv is not None:
                    adv = torch.tensor(adv).float().cuda()

            elif paradict['attack_method'] == 'hsja':
                if paradict['maxquery']<=100:
                    max_num_evals = 30 
                    print(f"max_num_evals:{max_num_evals}")
                elif paradict['maxquery']<=500:
                    max_num_evals = 180
                else:
                    max_num_evals = 100
                    #max_num_evals = 500
                adv,succ,query,_,_ = hsja(net, paradict['linf_con'],batch[0].detach().cpu().numpy(), constraint = paradict['constraint'],num_iterations = paradict["maxquery"],max_num_evals=max_num_evals)   
                if adv is not None: 
                    adv = torch.tensor(adv).float().cuda()

            else:
                adv,succ,query = attack_algo.forward(batch, label, target_label,imgidx=iteridx)



    if succ and query<=paradict["maxquery"]:
        advl2 = torch.norm(adv-batch)
        advlinf = torch.norm(adv-batch,p=float('inf'))

        succ_list.append(iteridx)
        query_list.append(int(query))
        l2_list.append(float(advl2))
        linf_list.append(float(advlinf))
    else:
        advl2 = None
        advlinf = None
    all_q_list.append(int(query))
    if advl2 is not None:
        all_l2_list.append(float(advl2))
        all_linf_list.append(float(advlinf))
    print('Img:{} succ:{} query:{} '.format(str(iTotal+1),succ,query))
    
    
    iTotal+=1 
    iteridx+=1
    asr = len(succ_list)/iTotal
    avg_q = np.mean(np.asarray(query_list))
    median_q = np.median(np.asarray(query_list))

    avg_l2 = np.mean(np.asarray(l2_list))
    avg_linf = np.mean(np.asarray(linf_list))

    median_l2 = np.median(np.asarray(l2_list))
    median_linf = np.median(np.asarray(linf_list))

    all_avg_q = np.mean(np.asarray(all_q_list))
    all_median_q = np.median(np.asarray(all_q_list))

    all_avg_l2 = np.mean(np.asarray(all_l2_list))
    all_avg_linf = np.mean(np.asarray(all_linf_list))

    all_median_l2 = np.median(np.asarray(all_l2_list))
    all_median_linf = np.median(np.asarray(all_linf_list))


    if iTotal%10==0:
        print("Model ACC:{:.4f}".format(iTotal/(iTotal+fail)))
        print("Succ rate:{:.4f}".format(asr))
        print("Succ Avg.query:{:.4f}".format(avg_q))
        print("Succ Avg.l2_list:{:.4f}".format(avg_l2))
        print("Succ Avg.linf_list:{:.4f}".format(avg_linf))    

        print("Succ Median.query:{:.4f}".format(median_q))
        print("Succ Median.l2_list:{:.4f}".format(median_l2))
        print("Succ Median.linf_list:{:.4f}".format(median_linf))  

        print("All Avg.query:{:.4f}".format(all_avg_q))
        print("All Avg.l2_list:{:.4f}".format(all_avg_l2))
        print("All Avg.linf_list:{:.4f}".format(all_avg_linf))    

        print("All Median.query:{:.4f}".format(all_median_q))
        print("All Median.l2_list:{:.4f}".format(all_median_l2))
        print("All Median.linf_list:{:.4f}".format(all_median_linf))          
        
    if iTotal>=200:
        break

print("=======================================================================================")
print("Model ACC:{:.4f}".format(iTotal/(iTotal+fail)))
print("Succ rate:{:.4f}".format(asr))
print("Succ Avg.query:{:.4f}".format(avg_q))
print("Succ Avg.l2_list:{:.4f}".format(avg_l2))
print("Succ Avg.linf_list:{:.4f}".format(avg_linf))    

print("Succ Median .query:{:.4f}".format(median_q))
print("Succ Median.l2_list:{:.4f}".format(median_l2))
print("Succ Median.linf_list:{:.4f}".format(median_linf))   