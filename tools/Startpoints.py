import torch
import numpy as np

def atleast_kdim(x, ndim):
    shape = x.shape + (1,) * (ndim - len(x.shape))
    return x.reshape(shape)

def _binary_search(model,labely,originals,perturbed: torch.Tensor, boost= False) -> torch.Tensor:
    # Choose upper thresholds in binary search based on constraint.
    highs = torch.ones(len(perturbed)).to(perturbed.device)
    d = np.prod(perturbed.shape[1:])
    lows = torch.zeros_like(highs)


    # use this variable to check when mids stays constant and the BS has converged
    iteration = 0
    #while torch.any(highs - lows > thresholds) and iteration < self._BS_max_iteration:
    while iteration<10:
        iteration += 1
        mids = (lows + highs) / 2
        epsilon = atleast_kdim(mids, len(originals.shape))

        mids_perturbed =  (1.0 - epsilon) * originals + epsilon * perturbed
        
        pre = model(mids_perturbed)
        is_adversarial_ = (torch.argmax(pre)!=labely)

        highs = torch.where(is_adversarial_, mids, highs)
        lows = torch.where(is_adversarial_, lows, mids)

    return  (1.0 - highs) * originals + highs * perturbed

@torch.no_grad()
def GaussianInit(filename,model, original_image_x, label_y):

    original_image_x=original_image_x.cuda()
    init = original_image_x.clone()
    p =torch.argmax(model(original_image_x))

    while p == label_y:
        init =torch.clamp(original_image_x+0.5*torch.randn_like(original_image_x),0,1)
        
        p = torch.argmax(model(init))

    advimg = _binary_search(model,label_y,original_image_x,init)
    query=10
    with torch.no_grad():
        pre = torch.argmax(model(advimg.cuda()))
    np.save(f'../data/GaussianInitAE/{filename.split(".")[0]}.npy',advimg.detach().cpu().numpy())
   