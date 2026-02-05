import torch
from SurFreeutils.utils import atleast_kdim


def get_init_with_noise(model, X, y,maxquery=1e5):
    init = X.clone()
    p = model(X).argmax(1)
    query = 0
    while any(p == y):
        init = torch.where(
            atleast_kdim(p == y, len(X.shape)), 
            (X + 0.5*torch.randn_like(X)).clip(0, 1), 
            init)
        p = model(init).argmax(1)
        query +=1
        if query > maxquery:
            break
    return init,query
