import numpy as np
import torch,math
import os, torch
import sys
import time
#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class RayS(object):
    def __init__(self, model, epsilon=0.031, order=np.inf,freqSearch=False,filterhis=True,analyzeGrad=False,netname="",binaryAnalyze=0):
        self.model = model
        self.ord = order
        self.epsilon = epsilon
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.queries = None
        self.npop = 1 
        self.nchannel = 3
        self.step_p = 0.01
        self.filterhis = filterhis
        self.filtertype="equal"
        self.analyzeGrad = analyzeGrad
        self.netname = netname
        self.binaryAnalyze = binaryAnalyze


    def get_xadv(self, x, v, d, lb=0., ub=1.):
        if isinstance(d, int):
            d = torch.tensor(d).repeat(len(x)).cuda()
        if self.ord==2:
            out = x + d * v
        elif self.ord == np.inf:
            out = x + d.view(len(x), 1, 1, 1) * v
        out = torch.clamp(out, lb, ub)
        return out




    @torch.no_grad()
    def attack_hard_label(self, x, y, target=None,refimg=None, query_limit=10000, seed=None,filename="",tracker=None,\
                          blacklight_threshold=25,\
                                 blacklight_count=0,blacklight_first_detect=0):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x, y): original image
        """
        shape = list(x.shape)
        dim = np.prod(shape[1:])
        if seed is not None:
            np.random.seed(seed)

        # init variables
        self.queries = torch.zeros_like(y).cuda()
        stop_queries = self.queries.clone()
        self.sgn_t = torch.sign(torch.ones(shape)).cuda()
        self.d_t = torch.ones_like(y).float().fill_(float("Inf")).cuda()
        working_ind = (self.d_t > self.epsilon).nonzero().flatten()

        
        dist = self.d_t.clone()
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)

        block_level = 0
        block_ind = 0
                
        signhist=self.sgn_t.clone()
        signhist_norm = self.sgn_t.clone()/torch.norm(self.sgn_t)
        res=[]
        bestdit = []
        #hisboundary = self.x_final.clone()
        to_update_ind =None
        gtgrad = None
        if self.analyzeGrad:
            if self.netname == "Resnet50":
                gtfolder=f"../data/GradSignSimilarity/FGSM_CE_origianlx"
                gtgradpath = f"../data/GradSignSimilarity/FGSM_CE_origianlx/{filename}.pth"
            else:
                gtfolder=f"../data/GradSignSimilarity/FGSM_CE_origianlx_{self.netname}"
                gtgradpath = f"../data/GradSignSimilarity/FGSM_CE_origianlx_{self.netname}/{filename}.pth"
            
            if not os.path.exists(gtfolder):
                os.makedirs(gtfolder)
                print(f"create folder:{gtfolder}")
            if not os.path.exists(gtgradpath) and self.netname != "Resnet50" and self.netname != "vit":
                print(f"gtgradpath:{gtgradpath} not exists")
                with torch.enable_grad():
                    images = x.detach().clone()
                    images.requires_grad = True
                    outputs = self.model(images)
                    celos = torch.nn.CrossEntropyLoss(reduction='none')
                    # Calculate loss
                    cost = celos(outputs, y)
                    #cost = torch.norm(startpoint-images,p=np.inf)-celos(startpoint, labels)

                    # Update adversarial images
                    gtgrad = torch.autograd.grad(cost, images,
                                            retain_graph=False, create_graph=False)[0]

                torch.save(gtgrad.detach().cpu().numpy(),gtgradpath) 
            else:   
                gtgrad = torch.tensor(torch.load(gtgradpath))
        cossimlist,querylist = [],[]

        for i in range(int(query_limit)):
            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)
            #if to_update_ind is not None and to_update_ind==1:
                #print(f"query:{int(self.queries)},width:{end-start}")
            valid_mask = (self.queries < query_limit) 
            attempt = self.sgn_t.clone().view(shape[0], dim)
                
            attempt[valid_mask.nonzero().flatten(), start:end] *= -1.
            attempt = attempt.view(shape)
            
            if self.filterhis:
                if self.filtertype=='equal':
                    condition =  torch.any(torch.all(attempt.flatten(start_dim=1).repeat(signhist.shape[0],1)==signhist.flatten(start_dim=1),dim=1))
                elif self.filtertype=='angle':
                    attempt_nrom = attempt.clone()/torch.norm(attempt)
                    
                    cosa = torch.einsum('ijkl,ijkl->i',[attempt_nrom.repeat(signhist_norm.shape[0],1,1,1),signhist_norm])
                    print(cosa)
                    #condition = torch.max(torch.abs(cosa))>torch.cos(torch.tensor([85])*math.pi/180).cuda()
                    condition=(cosa>=1)
                if condition:    
                    block_ind += 1
                    if block_ind == 2 ** block_level or end == dim:
                        block_level += 1
                        block_ind = 0                    
                    continue
                else:
                    signhist = torch.cat([signhist,attempt],dim=0)
                    if self.filtertype=='angle':
                        signhist_norm = torch.cat([signhist_norm,attempt_nrom],dim=0)
            prequery = self.queries[0].item()
            initial_succ_mask,to_update_ind,blacklight_count,blacklight_first_detect = self.binary_search(x, y, target, attempt, valid_mask,tracker=tracker,blacklight_threshold=blacklight_threshold,\
                                 blacklight_count=blacklight_count,blacklight_first_detect=blacklight_first_detect)
            if self.binaryAnalyze==2:
                diffq = self.queries[0].item()-prequery
                return self.x_final, (dist <= self.epsilon),stop_queries ,blacklight_count,blacklight_first_detect,diffq
            if to_update_ind>0 and gtgrad is not None:
                #cossim = float(torch.cosine_similarity(torch.sign(attempt.cpu()-x.cpu()).flatten(start_dim=1).cpu(),torch.sign(gtgrad).flatten(start_dim=1).cpu(),dim=1))
                #cossimlist.append(cossim)
                cossim = float(torch.cosine_similarity(attempt.cpu().flatten(start_dim=1).cpu(),torch.sign(gtgrad).flatten(start_dim=1).cpu(),dim=1))

                cossimlist.append([cossim,self.queries[0].item()])
                querytime = int(self.queries[0])
                querylist.append(querytime)
            if tracker is not None and blacklight_count>0:
                if self.analyzeGrad:
                    return self.x_final, (dist <= self.epsilon),stop_queries ,blacklight_count,blacklight_first_detect,cossimlist
                else:
                    return self.x_final, (dist <= self.epsilon),stop_queries ,blacklight_count,blacklight_first_detect
            
            #res.append([list(torch.abs(cosa).cpu().numpy()),int(initial_succ_mask),int(to_update_ind)])
            #hisboundary = torch.cat([hisboundary,self.x_final],dim=0)
            #l2norm = torch.norm(hisboundary.flatten(start_dim=1)-x.flatten(start_dim=1),dim=1)
            #print(l2norm)
            #save_image(self.x_final,'test.png')
            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.norm((self.x_final - x).view(shape[0], -1), self.ord, 1)
            stop_queries[working_ind] = self.queries[working_ind]
            working_ind = (dist > self.epsilon).nonzero().flatten()

            #print(bestdit)
            if torch.sum(self.queries >= query_limit) == shape[0]:
                print('out of queries')
                break
            if dist<=self.epsilon:
                break
            # progress_bar(torch.min(self.queries.float()), query_limit,
            #              'd_t: %.4f | adbd: %.4f | queries: %.4f | rob acc: %.4f | iter: %d'
            #              % (torch.mean(self.d_t), torch.mean(dist), torch.mean(self.queries.float()),
            #                 len(working_ind) / len(x), i + 1))
 

        stop_queries = torch.clamp(stop_queries, 0, query_limit)
        #return self.x_final, stop_queries, dist, (dist <= self.epsilon)
        #np.save(f'{filename}.npy',res)
        if self.analyzeGrad:
            return self.x_final, (dist <= self.epsilon),stop_queries ,blacklight_count,blacklight_first_detect,cossimlist
        else:
            return self.x_final, (dist <= self.epsilon),stop_queries,blacklight_count,blacklight_first_detect 


    @torch.no_grad()
    def attack_hard_label_naive(self, x, y, target=None,refimg=None, query_limit=10000, seed=None,filename="",tracker=None,\
                          blacklight_threshold=25,\
                                 blacklight_count=0,blacklight_first_detect=0):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x, y): original image
        """
        shape = list(x.shape)
        dim = np.prod(shape[1:])
        if seed is not None:
            np.random.seed(seed)

        # init variables
        self.queries = torch.zeros_like(y).cuda()
        stop_queries = self.queries.clone()
        self.sgn_t = torch.sign(torch.ones(shape)).cuda()
        self.d_t = torch.ones_like(y).float().fill_(float("Inf")).cuda()
        working_ind = (self.d_t > self.epsilon).nonzero().flatten()

        
        dist = self.d_t.clone()
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)

        block_level = 0
        block_ind = 0
        signhist=self.sgn_t.clone()
        signhist_norm = self.sgn_t.clone()/torch.norm(self.sgn_t)
        res=[]
        bestdit = []
        #hisboundary = self.x_final.clone()
        to_update_ind =None
        gtgrad = None
        if self.analyzeGrad:
            if self.netname == "Resnet50":
                gtfolder=f"../data/GradSignSimilarity/FGSM_CE_origianlx"
                gtgradpath = f"../data/GradSignSimilarity/FGSM_CE_origianlx/{filename}.pth"
            #gtgradpath = f"../data/GradSignSimilarity/CWGrad/ourstd_bs4_v3_{self.netname}/{filename}.pth"
            else:
                gtfolder=f"../data/GradSignSimilarity/FGSM_CE_origianlx_{self.netname}"
                gtgradpath = f"../data/GradSignSimilarity/FGSM_CE_origianlx_{self.netname}/{filename}.pth"
            
            #gtgradpath = f"../data/GradSignSimilarity/FGSM_CE_origianlx_Densenet121/{filename}.pth"
            if not os.path.exists(gtfolder):
                os.makedirs(gtfolder)
                print(f"create folder:{gtfolder}")
            if not os.path.exists(gtgradpath) and self.netname != "Resnet50" and self.netname != "vit":
                print(f"gtgradpath:{gtgradpath} not exists")
                with torch.enable_grad():
                    images = x.detach().clone()
                    images.requires_grad = True
                    outputs = self.model(images)
                    celos = torch.nn.CrossEntropyLoss(reduction='none')
                    # Calculate loss
                    cost = celos(outputs, y)
                    #cost = torch.norm(startpoint-images,p=np.inf)-celos(startpoint, labels)

                    # Update adversarial images
                    gtgrad = torch.autograd.grad(cost, images,
                                            retain_graph=False, create_graph=False)[0]

                torch.save(gtgrad.detach().cpu().numpy(),gtgradpath) 
            else:   
                gtgrad = torch.tensor(torch.load(gtgradpath))
        cossimlist,querylist = [],[]
        block_level = 0
        block_num = dim
        block_size = 1
        for i in range(int(query_limit)):
            
            
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)
            #if to_update_ind is not None and to_update_ind==1:
                #print(f"query:{int(self.queries)},width:{end-start}")
            valid_mask = (self.queries < query_limit) 
            attempt = self.sgn_t.clone().view(shape[0], dim)
                
            
            attempt[valid_mask.nonzero().flatten(), start:end] *= -1.
            attempt = attempt.view(shape)
            
            prequery = self.queries[0].item()
            initial_succ_mask,to_update_ind,blacklight_count,blacklight_first_detect = self.binary_search(x, y, target, attempt, valid_mask,tracker=tracker,blacklight_threshold=blacklight_threshold,\
                                 blacklight_count=blacklight_count,blacklight_first_detect=blacklight_first_detect)
            if self.binaryAnalyze==2:
                diffq = self.queries[0].item()-prequery
                return self.x_final, (dist <= self.epsilon),stop_queries ,blacklight_count,blacklight_first_detect,diffq
            if to_update_ind>0 and gtgrad is not None:
                #cossim = float(torch.cosine_similarity(torch.sign(attempt.cpu()-x.cpu()).flatten(start_dim=1).cpu(),torch.sign(gtgrad).flatten(start_dim=1).cpu(),dim=1))
                #cossimlist.append(cossim)
                cossim = float(torch.cosine_similarity(attempt.cpu().flatten(start_dim=1).cpu(),torch.sign(gtgrad).flatten(start_dim=1).cpu(),dim=1))

                cossimlist.append([cossim,self.queries[0].item()])
                querytime = int(self.queries[0])
                querylist.append(querytime)
            #res.append([list(torch.abs(cosa).cpu().numpy()),int(initial_succ_mask),int(to_update_ind)])
            #hisboundary = torch.cat([hisboundary,self.x_final],dim=0)
            #l2norm = torch.norm(hisboundary.flatten(start_dim=1)-x.flatten(start_dim=1),dim=1)
            #print(l2norm)
            #save_image(self.x_final,'test.png')
            block_ind += 1
            #if block_ind == 2 ** block_level or end == dim:
            if  end == dim:
                block_ind = 0

            dist = torch.norm((self.x_final - x).view(shape[0], -1), self.ord, 1)
            stop_queries[working_ind] = self.queries[working_ind]
            working_ind = (dist > self.epsilon).nonzero().flatten()

            #print(bestdit)
            if torch.sum(self.queries >= query_limit) == shape[0]:
                print('out of queries')
                break
            if dist<=self.epsilon:
                break
            # progress_bar(torch.min(self.queries.float()), query_limit,
            #              'd_t: %.4f | adbd: %.4f | queries: %.4f | rob acc: %.4f | iter: %d'
            #              % (torch.mean(self.d_t), torch.mean(dist), torch.mean(self.queries.float()),
            #                 len(working_ind) / len(x), i + 1))
 

        stop_queries = torch.clamp(stop_queries, 0, query_limit)
        #return self.x_final, stop_queries, dist, (dist <= self.epsilon)
        #np.save(f'{filename}.npy',res)
        if self.analyzeGrad:
            return self.x_final, (dist <= self.epsilon),stop_queries ,blacklight_count,blacklight_first_detect,cossimlist
        else:
            return self.x_final, (dist <= self.epsilon),stop_queries,blacklight_count,blacklight_first_detect 

    # check whether solution is found
    @torch.no_grad()
    def search_succ(self, x, y, target, mask):
        self.queries[mask] += 1
        # if target:
        #     return self.model.predict_label(x[mask]) == target[mask]
        # else:
        #     return self.model.predict_label(x[mask]) != y[mask]
        if len(x[mask])==5:
            output = self.model(x[mask][0])
        else:
            output = self.model(x[mask])
        
        predict = torch.argmax(output,dim=1)
        if target>-1:
        #if target:
            return  predict== target[mask]
        else:
            return predict != y[mask]
    # binary search for decision boundary along sgn direction
    def binary_search(self, x, y, target, sgn, valid_mask, tol=1e-3,tracker=None,\
                          blacklight_threshold=25,\
                                 blacklight_count=0,blacklight_first_detect=0):
        sgn_norm = torch.norm(sgn.view(len(x), -1), 2, 1)#l2 norm, dim=1
        sgn_unit = sgn / sgn_norm.view(len(x), 1, 1, 1)

        d_start = torch.zeros_like(y).float().cuda()
        d_end = self.d_t.clone()

        initial_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target, valid_mask)
        if tracker is not None:
            match_num = tracker.add_img(self.get_xadv(x, sgn_unit, self.d_t)[0].detach().cpu().numpy())
            if match_num>blacklight_threshold:
                blacklight_count+=1
                if blacklight_first_detect ==0:
                    blacklight_first_detect = int(self.queries[0])
                return None,None,blacklight_count,blacklight_first_detect


        #print(f"initial_succ_mask:{int(initial_succ_mask)}")
        to_search_ind = valid_mask.nonzero().flatten()[initial_succ_mask]
        d_end[to_search_ind] = torch.min(self.d_t, sgn_norm)[to_search_ind]

        while len(to_search_ind) > 0:
            d_mid = (d_start + d_end) / 2.0
            search_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, d_mid), y, target, to_search_ind)
            if tracker is not None:
                match_num = tracker.add_img(self.get_xadv(x, sgn_unit, d_mid)[0].detach().cpu().numpy())
                if match_num>blacklight_threshold:
                    blacklight_count+=1
                    if blacklight_first_detect ==0:
                        blacklight_first_detect = int(self.queries[0])
                    return None,None,blacklight_count,blacklight_first_detect

            d_end[to_search_ind[search_succ_mask]] = d_mid[to_search_ind[search_succ_mask]]
            d_start[to_search_ind[~search_succ_mask]] = d_mid[to_search_ind[~search_succ_mask]]
            to_search_ind = to_search_ind[((d_end - d_start)[to_search_ind] > tol)]

        to_update_ind = (d_end < self.d_t).nonzero().flatten()
        #print(f"to_update_ind:{int(len(to_update_ind))}")
        if len(to_update_ind) > 0:
            self.d_t[to_update_ind] = d_end[to_update_ind]
            self.x_final[to_update_ind] = self.get_xadv(x, sgn_unit, d_end)[to_update_ind]
            self.sgn_t[to_update_ind] = sgn[to_update_ind]
        return initial_succ_mask,len(to_update_ind),None,None

    def binary_search_limit(self, x, y, target, sgn, valid_mask, tol=1e-3,qlimit=10):
        #sgn_norm = torch.norm(sgn.view(len(x), -1), 2, 1)#l2 norm, dim=1
        sgn_norm = torch.norm(sgn.reshape(len(x), -1), 2, 1)
        sgn_unit = sgn / sgn_norm.view(len(x), 1, 1, 1)

        d_start = torch.zeros_like(y).float().cuda()
        d_end = self.d_t.clone()

        to_ask = self.get_xadv(x, sgn_unit, self.d_t)
        initial_succ_mask = self.search_succ(to_ask, y, target, valid_mask)
        to_ask_dis = torch.norm(to_ask-x,p=self.ord)
        if initial_succ_mask==True and to_ask_dis<=self.epsilon:
            to_update_ind = (to_ask_dis < self.d_t).nonzero().flatten()
            self.d_t[to_update_ind] = d_end[to_update_ind]
            self.x_final[to_update_ind] = self.get_xadv(x, sgn_unit, d_end)[to_update_ind]
            self.sgn_t[to_update_ind] = sgn[to_update_ind]
            return initial_succ_mask,len(to_update_ind)
        #print(f"initial_succ_mask:{int(initial_succ_mask)}")
        to_search_ind = valid_mask.nonzero().flatten()[initial_succ_mask]
        d_end[to_search_ind] = torch.min(self.d_t, sgn_norm)[to_search_ind]
        bs = 0
        while len(to_search_ind) > 0 and bs<qlimit:
            d_mid = (d_start + d_end) / 2.0
            to_ask = self.get_xadv(x, sgn_unit, d_mid)
            search_succ_mask = self.search_succ(to_ask, y, target, to_search_ind)
            d_end[to_search_ind[search_succ_mask]] = d_mid[to_search_ind[search_succ_mask]]
            d_start[to_search_ind[~search_succ_mask]] = d_mid[to_search_ind[~search_succ_mask]]
            to_search_ind = to_search_ind[((d_end - d_start)[to_search_ind] > tol)]
            bs += 1
            to_ask_dis = torch.norm(to_ask-x,p=self.ord)
            if to_ask_dis<=self.epsilon and search_succ_mask:
                break
        to_update_ind = (d_end < self.d_t).nonzero().flatten()
        #print(f"to_update_ind:{int(len(to_update_ind))}")
        if len(to_update_ind) > 0:
            self.d_t[to_update_ind] = d_end[to_update_ind]
            self.x_final[to_update_ind] = self.get_xadv(x, sgn_unit, d_end)[to_update_ind]
            self.sgn_t[to_update_ind] = sgn[to_update_ind]
        return initial_succ_mask,len(to_update_ind)
    
    def __call__(self, data, label, target=None,refimg=None, query_limit=10000,filename=""):
        return self.attack_hard_label(data, label, target=target, refimg=refimg,query_limit=query_limit,filename=filename)