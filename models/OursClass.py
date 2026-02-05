
    
import copy
import random
import numpy as np
import torch
from datetime import datetime
from PIL import Image
from ..tools.DataTools import ADBEvaluate

##################################################################################################
#represent blocks of a picture


class Block:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.width = x2 - x1 + 1

    def cut_block(self, subnum=2):
        bs = []
        for i in range(subnum):
            bs.append(copy.deepcopy(self))
 
        for i in range(subnum):
            line1 = self.x1 + (i * (self.x2 - self.x1)) // subnum #d1
            line2 = self.x1 + ((i + 1) * (self.x2 - self.x1)) // subnum #d2
            bs[i].x1 = line1
            if i > 0:
                bs[i].x1 = line1 + 1
            bs[i].x2 = line2
            bs[i].width = bs[i].x2 - bs[i].x1 + 1
        return bs

#represent a perturbation direction
class V:
    def __init__(self,ablation, size_channel, size_x, size_y, v,adv_v=None,Rmax=None):
        self.ablation = ablation
        self.size_channel = size_channel
        self.size_x = size_x
        self.size_y = size_y
        self.pixnum = size_x * size_y * size_channel
        self.adv_v = [v for _ in range(self.pixnum)]
        self.score = 1.0
        if Rmax is not None:
            self.Rmax = Rmax
        else:
            self.Rmax = 1.0
        self.Rmin = 0.0

        list_temp = [-1, 1]
        if v == 0:
            for x in range(len(self.adv_v)):
                self.adv_v[x] = random.choice(list_temp)
        elif v == -1e5:#
            self.adv_v = adv_v 
        self.init_v = np.argsort(np.array(self.adv_v),kind='stable')#
        self.continue_subarr = self.getContinueSubstr(list(self.init_v))

    def split_block(self, blocki,subnum=2):
        idx = blocki.x1
        subblock = self.continue_subarr[idx]
        lens = len(subblock)
        if lens >= subnum:
            self.continue_subarr[idx] = subblock[:lens//2]
            self.continue_subarr.insert(idx+1,subblock[lens//2:])
            
            block1 = Block(blocki.x1, min(blocki.x1 + 2,len(self.continue_subarr)-1))
            return block1 ,True
        else:
            return blocki,False


    def getContinueSubstr(self,lista):
        res=[]
        curr_group =0
        for i in lista:
            if len(res)==0:
                res.append([i])
            else:
                if i==res[curr_group][-1]+1:
                    res[curr_group].append(i)
                else:
                    curr_group+=1
                    res.append([i])
        return res 

    #reverse blocks of a perturbation direction to generate new directions
    def reverse_v(self, block):
        if self.ablation==0 or (self.ablation>1 and self.ablation<10):
            blocks = self.continue_subarr[block.x1:block.x2+1]
            newarr = np.array(self.adv_v)
            for blocki in blocks:   
                newarr[blocki] *= -1
            self.adv_v = list(newarr)
        elif self.ablation ==1 or self.ablation>10:
            for x in range(block.x1, block.x2 + 1):
                self.adv_v[x] *= -1
        else:
            print("error")
            raise TypeError  
           

    def advv_to_tensor(self):
        # initialize 
        three_d_list = [
            [[self.adv_v[channel * self.size_x * self.size_y + x * self.size_y + y]
              for y in range(self.size_y)]
             for x in range(self.size_x)]
            for channel in range(self.size_channel)]
        aim_np = np.array(three_d_list)
        perturbation = torch.tensor(aim_np)
        return perturbation
    


#represent Iterations
class Iter:
    def __init__(self, init_vbest, offspringN, iter_n=1,early_stop=False,tracker=None,paratype=10,useadba=1,api_type=None):
        self.offspringN = offspringN
        self.iter_n = iter_n
        self.offspringVs = [] #d1 d2
        self.early_stop = early_stop
        self.chosen_v = -1 #
        self.old_vbest = copy.deepcopy(init_vbest) # dbest
        for i in range(offspringN):

            self.offspringVs.append(copy.deepcopy(init_vbest))
            self.offspringVs[i].Rmax, self.offspringVs[i].Rmin = 1.0, 0.0
        self.blacklight_threshold = 25 
        self.blacklight_count = 0
        self.blacklight_first_detect = 0
        self.tracker = tracker
        self.hessian_batch= 10
        self.hessian_downsample = 4
        self.hessian_mu = 0.001
        self.adbafun = ADBEvaluate(PARA_TYPE=paratype)
        self.adb_interval = 5 
        print(f"self.adb_interval:{self.adb_interval}")
        self.useadba =useadba
        self.api_type = api_type
    #create a new generation


    def getTempFilename(self,image):
        timpstampnow = int(round(datetime.now().timestamp()))+random.randint(0,1000)
        savep = f"../data/tmp_{self.api_type}_ours/temp_{timpstampnow}.png"
        Image.fromarray(np.uint8(np.round(image[0].permute(1,2,0).detach().cpu().numpy()*255))).save(savep)

        return savep

    def mutation(self, model, original_image, label, aim_r, tolerance_binary_iters, blocks, binaryM,method,globalq=0,hisv=[],hisblock=[],hispre=[],gtgrad=None,nonzeroidx=None):
        query = 0
        self.dim = original_image.shape[1] * original_image.shape[2] * original_image.shape[3]
        cossim=[]
        for vi in range(self.offspringN):
            self.offspringVs[vi].reverse_v(blocks[vi])
            self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = self.old_vbest.Rmax, 0.0
        orioffspringlen = len(self.offspringVs)
       
        query_plus,hisv,hisblock,hispre =  self.compare_directions_usingADB(
            model, original_image, label, aim_r, tolerance_binary_iters, binaryM,method,globalq,hisv,hisblock,blocks,hispre,orioffspringlen)
        query = query +query_plus

        for vi in range(orioffspringlen): #initialize directions
            self.offspringVs[vi].reverse_v(blocks[vi])
        if self.chosen_v >= 0:
            if self.chosen_v == orioffspringlen:
                self.old_vbest.adv_v = list(self.offspringVs[self.chosen_v].adv_v)
                self.old_vbest.Rmax, self.old_vbest.Rmin = (
                self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)

                for vi in range(orioffspringlen):
                    self.offspringVs[vi].adv_v = list(self.offspringVs[self.chosen_v].adv_v)
                    self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = (
                        self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)
                    
                self.offspringVs.pop(orioffspringlen)

            else:
                self.old_vbest.reverse_v(blocks[self.chosen_v])
                self.old_vbest.Rmax, self.old_vbest.Rmin = (
                    self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)
                for vi in range(orioffspringlen):
                    self.offspringVs[vi].reverse_v(blocks[self.chosen_v])
                    self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = (
                        self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)
                
                if len(self.offspringVs)== (orioffspringlen+1):
                    self.offspringVs.pop(orioffspringlen)
 
        self.iter_n = self.iter_n + 1
        return query,hisv,hisblock,hispre



    def compare_directions_usingADB(self, model, original_image, label, aim_r, maxIters, binaryM,method,globalq=0,hisv=[],hisblock=[],blocks=[],hispre=[],orioffspringlen=2):
        perturbations = [] #d1 d2
        perturbed_images = [] #= x+ADB*d
        predicted = [] #= F(x+ADB*d)
        query = 0
        succV = []
        self.chosen_v = -1
        
        for i in range(len(self.offspringVs)):
            perturbations.append(self.offspringVs[i].advv_to_tensor().cuda())
            perturbed_images.append(torch.clamp(original_image.cuda() +
                                                self.old_vbest.Rmax * perturbations[i], 0.0, 1.0))
            if self.api_type in ["baidu","tencent","google","imagga"]:
                tmpsavep = self.getTempFilename(perturbed_images[i])
            if self.api_type=="baidu":
                predictLabel = model.detect_labels(tmpsavep,save=False)
            elif self.api_type=="tencent":
                _,predictLabel,_ = model.detect_labels(tmpsavep,query+1,save=False)
            elif self.api_type=="google":
                predictLabel,score =  model.predict_label(tmpsavep,tmpsavep.split(".")[0]+"_res.json")
            elif self.api_type=="imagga":
                predictLabel = model.predict_label(perturbed_images[i])
            elif self.api_type=="standard":
                predictLabel = model.predict_label(perturbed_images[i].cpu())
            elif self.api_type=="clip":
                predictLabel = torch.argmax(model(perturbed_images[i]).cpu())
            predicted.append(predictLabel)

            if self.tracker is not None:
                match_num = self.tracker.add_img(perturbed_images[i][0].detach().cpu().numpy())
                if match_num>self.blacklight_threshold:
                    self.blacklight_count+=1
                    if self.blacklight_first_detect ==0:
                        self.blacklight_first_detect = globalq+query + 1
            query = query + 1
            if(not model.ismatchHard(predictLabel) and (self.api_type=="baidu" or self.api_type=="tencent")) \
                or ((self.api_type=="google" or self.api_type=="standard") and predicted[i] != label)\
                    or (self.api_type=="imagga" and model.compare_label(predicted[i]))\
                        or (self.api_type=="clip" and predicted[i]!=int(label)):
                succV.append(i)
                self.offspringVs[i].Rmax = self.old_vbest.Rmax
                self.chosen_v = i
                # if self.early_stop:
                #     break
            else:
                self.offspringVs[i].Rmin = self.old_vbest.Rmax

        if len(succV) == 0:
            self.chosen_v = -1
            return query,hisv,hisblock,hispre

        elif len(succV) == 1:

            if len(hisv)>0:
                succV.append(orioffspringlen)
                perturbations.append(hisv[0].advv_to_tensor().cuda())
                perturbed_images.append(torch.clamp(original_image.cuda() +
                                    self.old_vbest.Rmax * perturbations[-1], 0.0, 1.0))
                predicted.append(hispre[0])
                self.offspringVs.append(hisv[0])

                hisv.pop(0)
                hispre.pop(0)
            else:
                hisv.append(copy.deepcopy(self.offspringVs[succV[0]]))
                self.chosen_v = succV[0]
                hispre.append(predicted[succV[0]])
                return query,hisv,hisblock,hispre
        

        low, high = 0, self.old_vbest.Rmax
        for ite in range(0, maxIters): #conmaration loop
            if self.useadba==1:
                ADB = self.adbafun.next_ADB(low, high, aim_r, self.old_vbest.Rmax, binaryM,method=method)
            else:
                ADB = high-(high-low)/self.adb_interval
            succVtemp = copy.deepcopy(succV)
            vi = 0
            while vi < len(succVtemp):
                perturbed_images[succVtemp[vi]] = torch.clamp(
                    original_image.cuda() + ADB * perturbations[succVtemp[vi]], 0.0, 1.0)
                if self.api_type in ["baidu","tencent","google","imagga"]:
                    tmpsavep = self.getTempFilename(perturbed_images[succVtemp[vi]])
                if self.api_type=="baidu":
                    predictLabel = model.detect_labels(tmpsavep,save=False)
                elif self.api_type=="tencent":
                    _,predictLabel,_ = model.detect_labels(tmpsavep,query+1,save=False)
                elif self.api_type=="google":
                    predictLabel,score =  model.predict_label(tmpsavep,tmpsavep.split(".")[0]+"_res.json")
                elif self.api_type=="imagga":
                    predictLabel = model.predict_label(perturbed_images[succVtemp[vi]])
                elif self.api_type=="standard":
                    predictLabel = model.predict_label(perturbed_images[succVtemp[vi]].cpu())
                elif self.api_type=="clip":
                    predictLabel = torch.argmax(model(perturbed_images[succVtemp[vi]]).cpu())
                predicted[succVtemp[vi]] = predictLabel
                query = query + 1
                if(not model.ismatchHard(predictLabel) and (self.api_type=="baidu" or self.api_type=="tencent")) \
                    or ((self.api_type=="google" or self.api_type=="standard") and predicted[succVtemp[vi]] != label)\
                        or (self.api_type=="imagga" and model.compare_label(predicted[succVtemp[vi]]))\
                        or (self.api_type=="clip" and predicted[succVtemp[vi]]!=int(label)):
                    self.offspringVs[succVtemp[vi]].Rmax = ADB
                    self.chosen_v = succVtemp[vi]
                    if self.offspringVs[succVtemp[vi]].Rmax <= aim_r:
                        self.chosen_v = succVtemp[vi]
                        return query,hisv,hisblock,hispre
                    vi = vi + 1
                else:
                    self.offspringVs[succVtemp[vi]].Rmin = ADB
                    succVtemp.pop(vi)

            if len(succVtemp) == 0:
                low = ADB
            elif len(succVtemp) == 1:
                self.chosen_v = succVtemp[0]
                return query,hisv,hisblock,hispre
            elif len(succVtemp) >= 2:
                high = ADB
                succV = succVtemp
            if ite >= 4 and high - low <= 0.0002:
                break

        self.chosen_v = succV[0] 
        return query,hisv,hisblock,hispre
    

    
def findOrder(matrix,rows=8,columns=8):

    result = [0]*(rows*columns)
    result[0] = matrix[0][0]
    k = 1
    i = j = 0
    while(k < rows*columns):
        while i >= 1 and j < rows-1:
            i -= 1
            j += 1
            result[k] = matrix[i][j]
            k += 1
        if j < rows-1:
            j += 1
            result[k] = matrix[i][j]
            k += 1
        elif i < columns-1:
            i += 1
            result[k] = matrix[i][j]
            k += 1
        while i < columns-1 and j >= 1:
            i += 1
            j -= 1
            result[k] = matrix[i][j]
            k += 1
        if i < columns-1:
            i += 1
            result[k] = matrix[i][j]
            k += 1
        elif j < rows-1:
            j += 1
            result[k] = matrix[i][j]
            k += 1
    return result


def getZigzagMeanStd(y_pristine_adv_diff):
    zigzaglist =[]
    r,c = y_pristine_adv_diff.shape[1],y_pristine_adv_diff.shape[2]
    for blocki in range(0,y_pristine_adv_diff.shape[0]):
        block_content = y_pristine_adv_diff[blocki]
        #zigzaglist.append(findOrder(block_content.cpu().numpy(),rows=r,columns=c))
        tmp = findOrder(block_content,rows=r,columns=c)
        zigzaglist.append(torch.stack(tmp))
 
    # y_pristine_adv_diff_mean = np.mean(np.array(zigzaglist),axis=0)
    # y_pristine_adv_diff_std = np.std(np.array(zigzaglist),axis=0)
    y_pristine_adv_diff_mean = torch.mean(torch.stack(zigzaglist),dim=0)
    y_pristine_adv_diff_std = torch.std(torch.stack(zigzaglist),dim=0)

    return y_pristine_adv_diff_mean,y_pristine_adv_diff_std,zigzaglist


def getzigzagcor(index,rows=8,columns=8):
    if index==0:
        return 0,0
    k = 1
    i = 0
    j = 0
    rows = rows
    columns = columns
    flag = False
    while(k < rows*columns):
        while i >= 1 and j < rows-1:
            i -= 1
            j += 1
            if k==index:
                flag = True
                break
            #matrix[i][j] = result[k]
            k += 1
        if flag:
            break
        if j < rows-1:
            j += 1
            if k==index:
                flag = True
                break            
            #matrix[i][j] = result[k] 
            k += 1
        elif i < columns-1:
            i += 1
            if k==index:
                flag = True
                break    
            #matrix[i][j] = result[k]
            k += 1
        while i < columns-1 and j >= 1:
            i += 1
            j -= 1
            if k==index:
                flag = True
                break    
            #matrix[i][j] = result[k]
            k += 1
        if flag:
            break
        if i < columns-1:
            i += 1
            if k==index:
                flag = True
                break   
            #matrix[i][j] = result[k]
            k += 1
        elif j < rows-1:
            j += 1
            if k==index:
                flag = True
                break   
            #matrix[i][j] = result[k]
            k += 1
    return i,j
@torch.no_grad()
def getNewd(ycbcr,n_block,stds,image,npop,nchannel,step_p,diffj,ord,initmu=0,\
            initystd=0,initcbstd=0,initcrstd=0,blocksize=8,init="",initvariables=[],freqratio=1,color=0,returnColorimg=False):
    if init =="":
        y_std,cb_std,cr_std = stds[0]+initystd,stds[1]+initcbstd,stds[2]+initcrstd
    else:
        y_variance,cb_variance,cr_variance = initvariables

    n_block = int((image.shape[2]/blocksize)**2)
    mu = torch.zeros(1,3,n_block,blocksize,blocksize)
    mu[:,:,:,0,0] += initmu
    modifys = torch.zeros(npop,nchannel,n_block,blocksize,blocksize)
    repeat_num = int(n_block//n_block)

    color=0
    stepp=1 
    
    step_p_y = stepp #[block_size:1:1e4;2: 100;4:1;8:0.01]
    #for color
    step_p_cb_color = stepp*100 
    step_p_cr_color = stepp*100
    step_p_cb = 0
    step_p_cr = 0
    #perturb_range = int(min(freqratio,blocksize**2))
    perturb_range = blocksize**2
    print(f"perturb_channel end at:{perturb_range},step_p_y:{step_p_y},step_p_cb:{step_p_cb},step_p_cr:{step_p_cr}")
    for i in range(perturb_range):
        zigzagi,zigzagj = getzigzagcor(i,rows=blocksize,columns=blocksize)
        mu_z = torch.randn((npop,nchannel,n_block))
        if not init=="":
            stdbias = torch.cat([(mu_z[:,0,:]*(y_variance[i])).unsqueeze(1),\
                                            (mu_z[:,1,:]*(cb_variance[i])).unsqueeze(1),\
                                            (mu_z[:,2,:]*(cr_variance[i])).unsqueeze(1)],dim=1)
        else:   
            stdbias = torch.cat([(mu_z[:,0,:]*(y_std[i]**2)).unsqueeze(1),\
                                (mu_z[:,1,:]*(cb_std[i]**2)).unsqueeze(1),\
                                (mu_z[:,2,:]*(cr_std[i]**2)).unsqueeze(1)],dim=1)
        
        modify = mu.repeat(npop,1,1,1,1)[:,:,:,zigzagi,zigzagj]+stdbias
        
        modifys[:,:,:,zigzagi,zigzagj] = modify

    modifys[:,0,:,:,:] *= step_p_y

    modifys_color = modifys.clone()

    modifys[:,1,:,:,:] *= step_p_cb
    modifys[:,2,:,:,:] *= step_p_cr

    modifys_color[:,1,:,:,:] *= step_p_cb_color
    modifys_color[:,2,:,:,:] *= step_p_cr_color
    
    ycbcr_pert = ycbcr.clone()+ modifys.repeat(1,1,repeat_num,1,1)
    
    adv_images = diffj.rec(ycbcr_pert[:,0],ycbcr_pert[:,1],ycbcr_pert[:,2],image.shape[2],image.shape[3]) 

    adv_images = adv_images.cuda()
    
    adv_images = torch.clamp(adv_images,0,1)  

    perturb_pixel =  adv_images-image 
    dis = torch.norm(perturb_pixel,p=ord)
    newd = perturb_pixel/torch.norm(perturb_pixel)


    ycbcr_pert_color = ycbcr.clone()+ modifys_color.repeat(1,1,repeat_num,1,1)        
    adv_images_color = diffj.rec(ycbcr_pert_color[:,0],ycbcr_pert_color[:,1],ycbcr_pert_color[:,2],image.shape[2],image.shape[3]) 
    adv_images_color = adv_images_color.cuda()
    adv_images_color = torch.clamp(adv_images_color,0,1)  
    perturb_pixel_color =  adv_images_color-image 
    dis_color = torch.norm(perturb_pixel_color,p=ord)
    newd_color = perturb_pixel_color/torch.norm(perturb_pixel_color)

    if returnColorimg:
        return newd,adv_images,dis,newd_color,adv_images_color
    else:
        return newd,adv_images,dis,newd_color
    



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
    #gtgradsign_avgppflatten=(gtgradsign_avgppflatten+1)/2
    #Image.fromarray(np.uint8(np.round((gtgradsign_avgppflatten[0].permute(1,2,0).numpy()*255)))).save('test.png')
    return gtgradsign_avgppflatten