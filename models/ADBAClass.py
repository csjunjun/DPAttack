import copy
import random
import numpy as np
import torch
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
    def __init__(self, size_channel, size_x, size_y, v,adv_v=None):
        self.size_channel = size_channel
        self.size_x = size_x
        self.size_y = size_y
        self.pixnum = size_x * size_y * size_channel
        self.adv_v = [v for _ in range(self.pixnum)]
        self.score = 1.0
        self.Rmax = 1.0
        #self.Rmax = 0.8
        #print(f"init V Rmax:{self.Rmax}")
        self.Rmin = 0.0

        list_temp = [-1, 1]
        for x in range(len(self.adv_v)):
            self.adv_v[x] = random.choice(list_temp)
        

    #reverse blocks of a perturbation direction to generate new directions
    def reverse_v(self, block):
        for x in range(block.x1, block.x2 + 1):
            self.adv_v[x] *= -1

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
        self.blacklight_threshold = 25 
        self.blacklight_count = 0
        self.blacklight_first_detect = 0
        self.tracker = tracker
        self.adbafun = ADBEvaluate(PARA_TYPE=PARA_TYPE)
        self.adb_interval = 5 
        self.useadba =useadba

        print(f"self.adb_interval:{self.adb_interval}")
    #create a new generation
    def mutation(self, model, original_image, label, aim_r, tolerance_binary_iters, blocks, binaryM,method,globalq=0,adaptive=0,gtgrad=None,cossim=[]):
        query = 0
        for vi in range(self.offspringN):
            self.offspringVs[vi].reverse_v(blocks[vi])
            self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = self.old_vbest.Rmax, 0.0

        query_plus,cossim = self.compare_directions_usingADB(
            model, original_image, label, aim_r, tolerance_binary_iters, binaryM,method,globalq,adaptive=adaptive,gtgrad=gtgrad,cossim=cossim)

        query = query + query_plus
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
        return query,cossim

    def compare_directions_usingADB(self, model, original_image, label, aim_r, maxIters,\
                                     binaryM,method,globalq=0,paras=None,adaptive=0,gtgrad=None,cossim=[]):
        perturbations = [] #d1 d2
        perturbed_images = [] 
        predicted = [] #= 
        query = 0
        succV = []
        self.chosen_v = -1
        #initialize directions
        for i in range(len(self.offspringVs)):
            perturbations.append(self.offspringVs[i].advv_to_tensor().cuda())
            perturbed_images.append(torch.clamp(original_image.cuda() +
                                            self.old_vbest.Rmax * perturbations[i], 0.0, 1.0))
            predicted.append(model.predict_label(perturbed_images[i]).cpu())
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
            else:
                self.offspringVs[i].Rmin = self.old_vbest.Rmax

        if len(succV) == 0:
            self.chosen_v = -1
            return query,cossim
        elif len(succV) == 1:
            self.chosen_v = succV[0]
            return query,cossim

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
                predicted[succVtemp[vi]] = model.predict_label(perturbed_images[succVtemp[vi]]).cpu()
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
                    if self.offspringVs[succVtemp[vi]].Rmax <= aim_r:
                        self.chosen_v = succVtemp[vi]
                        if gtgrad is not None:
                            cossim=[float(torch.cosine_similarity(torch.tensor(self.offspringVs[succVtemp[vi]].adv_v).unsqueeze(0),torch.sign(gtgrad).cpu().flatten(start_dim=1)))]
                        return query,cossim
                    vi = vi + 1
                else:
                    self.offspringVs[succVtemp[vi]].Rmin = ADB
                    succVtemp.pop(vi)

            if len(succVtemp) == 0:
                low = ADB
            elif len(succVtemp) == 1:
                self.chosen_v = succVtemp[0]
                if gtgrad is not None:
                    cossim=[float(torch.cosine_similarity(torch.tensor(self.offspringVs[self.chosen_v].adv_v).unsqueeze(0),torch.sign(gtgrad).cpu().flatten(start_dim=1)))]
                return query,cossim
            elif len(succVtemp) >= 2:
                high = ADB
                succV = succVtemp

            if ite >= 4 and high - low <= 0.0002:
                break

        #d1 and d2 are close, just return d1
        self.chosen_v = succV[0]
        if gtgrad is not None:
            cossim=[float(torch.cosine_similarity(torch.tensor(self.offspringVs[self.chosen_v].adv_v).unsqueeze(0),torch.sign(gtgrad).cpu().flatten(start_dim=1)))]
        return query,cossim
