import torch.nn.functional as F

def mse_recon_loss_fn(x_hat, x, logging: bool=True, postprocessing=F.sigmoid):
    if logging:  
        return F.mse_loss(postprocessing(x_hat), x, reduction='sum').div(x.shape[0])
    return F.mse_loss(postprocessing(x_hat), x)

def bce_recon_loss_fn(x_hat, x, logging: bool=True, logits: bool=True): 
    if logits: 
        fn = F.binary_cross_entropy_with_logits
    else: 
        fn = F.binary_cross_entropy
    if logging: 
        return fn(x_hat, x, reduction='sum').div(x.shape[0])
    return fn(x_hat, x)