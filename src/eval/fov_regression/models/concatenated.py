import torch
import torch.nn as nn 

class ConcatModels(nn.Module):
    def __init__(self, repn_fn_first_model, second_model):
        super().__init__()
        self.repn_fn_first_model = repn_fn_first_model
        self.second_model = second_model

    def forward(self, x: torch.Tensor):
        return self.second_model(self.repn_fn_first_model(x))
    
    def repn_fn(self, x: torch.Tensor, key: str=None) -> torch.Tensor: 
        return self(x)