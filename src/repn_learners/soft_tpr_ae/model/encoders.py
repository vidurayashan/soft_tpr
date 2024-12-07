import torch 
import torch.nn as nn
from typing import List

from src.shared.components import MLP, View

class AblationEncoder(nn.Module): 
    def __init__(self, rep_dim: int, post_processor: MLP=None) -> None: 
        super().__init__() 
        self.post_processor = post_processor 
        self.rep_dim = rep_dim
        self.kwargs_for_loading = {'rep_dim': rep_dim}
        if post_processor is not None: 
            self.kwargs_for_loading = {**self.kwargs_for_loading, 'post_processor_kwargs': 
                post_processor.kwargs_for_loading}
    
    def forward(self, gt_factor_classes: torch.Tensor) -> torch.Tensor: 
        gt_factor_classes = gt_factor_classes.to(dtype=torch.float32)
        if self.post_processor is not None: 
            return {'rep': self.post_processor(gt_factor_classes.squeeze())}
        return {'rep': gt_factor_classes.squeeze()}
    
class GTEncoder(AblationEncoder): 
    def __init__(self, rep_dim: int) -> None: 
        super().__init__(rep_dim=rep_dim, post_processor=None)

class ModularMLP(nn.Module): 
    def __init__(self, rep_dim: int, n_modules: int) -> None: 
        super().__init__() 
        self.nets = nn.ParameterList([
                MLP(in_dim=1, out_dim=rep_dim//n_modules) for _ in range(n_modules)
        ])
        self.kwargs_for_loading = {'n_modules': n_modules}
    
    def forward(self, gt_factor_classes: torch.Tensor) -> torch.Tensor: 
        #print(f'********************************Gt factor classes {gt_factor_classes.shape}')
        gt_factor_classes = gt_factor_classes.to(dtype=torch.float32)
        stacked_output = [net(gt_factor_classes[:, i].unsqueeze(1)) for i, net in enumerate(self.nets)]
        #print(f'Returning item of shape {torch.concatenate(stacked_output, dim=1).shape}')
        return torch.concatenate(stacked_output, dim=1)

class ModularEncoder(AblationEncoder): 
    def __init__(self, rep_dim: int, n_roles: int) -> None: 
        super().__init__(rep_dim=n_roles*(rep_dim//n_roles), post_processor=ModularMLP(rep_dim=rep_dim, n_modules=n_roles))
        self.kwargs_for_loading = {'n_roles': n_roles, 
                                   'rep_dim': rep_dim}

class VQ_VAE_Encoder(nn.Module): 
    def __init__(self, filler_embed_dim: int, role_embed_dim: int, 
                 hidden_dims: List[int], nc: int=3) -> None: 
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), # (256, 8, 8)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True), 
            View(-1, 512*4*4),
            MLP(in_dim=512*4*4, hidden_dims=hidden_dims, out_dim=filler_embed_dim*role_embed_dim)
        )
        self.role_embed_dim = role_embed_dim 
        self.filler_embed_dim = filler_embed_dim
        self.rep_dim = role_embed_dim * filler_embed_dim
        self.kwargs_for_loading = {
            'filler_embed_dim': filler_embed_dim, 
            'role_embed_dim': role_embed_dim, 
            'hidden_dims': hidden_dims,
            'nc': nc
        }
    
    def forward(self, x: torch.Tensor):
        return self.net(x).view(-1, self.filler_embed_dim, self.role_embed_dim) 
