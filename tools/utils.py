import numpy as np
import torch,sys
from blacklight import InputTracker
def setSeed(seed=0):
    np.random.seed(seed) 
    torch.manual_seed(seed) 
   
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False       
    torch.backends.cudnn.deterministic = True



#show the results dynamically
def progress_bar(imgi, query, iter, Rnow):
    sys.stdout.write(f'\rImg{imgi} Query{query :.0f}\t Iter{iter :.0f}\t Rinf{Rnow:.4f}')
    sys.stdout.flush()


def get_tracker(query, window_size, hash_kept, roundto, step_size, workers):
    tracker = InputTracker(query, window_size, hash_kept, round=roundto, step_size=step_size, workers=workers)
    
    return tracker



def getCifar10Testdata():
    import torchvision.transforms as transforms
    import torchvision
    from torch.utils.data import DataLoader

    # Show images
    valcnt = 0
    acc = 0
    batch_size=10
    transform_test  = transforms.Compose([        
            transforms.ToTensor(),
            ])
    test_data  = torchvision.datasets.CIFAR10('../data/cifar10/', train=False, transform=transform_test, download=True)

    test_loader   = DataLoader(test_data,  batch_size=batch_size, shuffle=True, num_workers=1,
                            pin_memory=True, drop_last=True)

    total =0 
    filename ="../data/cifar10/val.txt"
    for idx, test_batch in enumerate(test_loader):
        
        test_secrets, labels  = test_batch
        test_secrets = test_secrets
        valcnt += len(test_secrets)


        labels = labels

        

        with open(filename,'a+') as f:
            for i in range(len(test_secrets)):
                total +=1
                f.write(f"Cifar10_{total}.npy "+str(labels[i].item())+'\n')
                np.save(f"../data/cifar10/val/Cifar10_{total}.npy",test_secrets[i].cpu().numpy())
        if total>1000:
            break

def get_pert_amp(amp,maxamp,freqmask,type="fft"):
    if type=="fft":
        if len(amp.shape)==4 or len(amp.shape)==3:
            freqmask = freqmask.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).cuda()
        elif len(amp.shape)==2:
            freqmask = freqmask.unsqueeze(0).unsqueeze(0).cuda()
        rn = torch.ones_like(amp).cuda()*maxamp
        amp = amp*(1-freqmask)+(amp+rn)*freqmask
        return amp
    elif type=="dct":
        freqmask = freqmask.cuda()
        rn_ = torch.zeros_like(amp).cuda()
        rn = torch.randn((rn_.shape[1])).cuda()*maxamp
        rn_[torch.where(freqmask>0)]=rn
        amp = amp*(1-freqmask)+(amp+rn_)*freqmask
        return amp
    elif type=="dctuni":
        freqmask = freqmask.cuda()
        rn_ = torch.ones_like(amp).cuda()*maxamp
        amp = amp*(1-freqmask)+(amp+rn_)*freqmask
        return amp


