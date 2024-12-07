import abc
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from typing import Dict

from src.shared.constants import *

class AbstractAE(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, gt_factor_classes: torch.Tensor) -> torch.Tensor:
        return

    @abc.abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return
    
    @abc.abstractmethod
    def repn_fn(self, gt_factor_classes: torch.Tensor) -> torch.Tensor: 
        return 

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, gt_factor_classes: torch.Tensor, recon_loss_fn) -> Dict:
        """model return (reconstructed_x, *)"""
        return
    
    @abc.abstractmethod 
    def make_state(self, x: torch.Tensor, x_hat: torch.Tensor, latent: torch.Tensor, **kwargs) -> Dict:
        return 
