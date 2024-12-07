import torch 
import torch.nn as nn
from typing import Dict 
from torch import linalg as LA 

from src.shared.constants import *
from src.repn_learners.soft_tpr_ae.utils import init_embeddings

class BaseTPREncoder(nn.Module): 
    def __init__(self, n_roles: int, n_fillers: int, role_embed_dim: int, 
                 filler_embed_dim: int, use_concatenated_rep: bool, 
                 lambdas_reg: Dict={ORTH_PENALTY_ROLE: 0, ORTH_PENALTY_FILLER: 0}, 
                 init_roles_orth: bool=False, 
                 init_fillers_orth: bool=False,
                 fixed_roles: bool=False) -> None: 
        super().__init__() 
        if fixed_roles: 
            init_roles_orth=True
        self.role_embeddings = nn.Embedding(num_embeddings=n_roles, embedding_dim=role_embed_dim).requires_grad_(not fixed_roles)
        self.filler_embeddings = nn.Embedding(num_embeddings=n_fillers, embedding_dim=filler_embed_dim)
        self.use_concatenated_rep = use_concatenated_rep
        self.lambdas_reg = lambdas_reg

        if self.use_concatenated_rep: 
            self.rep_dim = role_embed_dim * filler_embed_dim * n_roles
        else: 
            self.rep_dim = role_embed_dim * filler_embed_dim
        
        self.init_weights(init_fillers_orth=init_fillers_orth, init_roles_orth=init_roles_orth)
            
        self.kwargs_for_loading = {
            'n_roles': n_roles, 
            'n_fillers': n_fillers, 
            'role_embed_dim': role_embed_dim, 
            'filler_embed_dim': filler_embed_dim, 
            'use_concatenated_rep': self.use_concatenated_rep, 
            'lambdas_reg': self.lambdas_reg, 
            'init_roles_orth': False,  # will be overriden when model params loaded
            'init_fillers_orth': False,
            'fixed_roles': fixed_roles
        }
        
    def init_weights(self, init_fillers_orth: bool, init_roles_orth: bool): 
        init_embeddings(init_fillers_orth, self.filler_embeddings.weight)
        init_embeddings(init_roles_orth, self.role_embeddings.weight)
    
    def forward(self, batched_roles: torch.Tensor, batched_fillers: torch.Tensor) -> Dict: 
        """ 
        Args: 
            batched_roles (torch.Tensor) of dimension (N_{B}, N_{R}, D_{R})
            batched_fillers (torch.Tensor) of dimension (N_{B}, N_{R}, D_{F})
        Receives a batch of roles and corresponding bound fillers and produces TPR bindings 
        The binding of filler f_{i} \in \mathbb{R}^{D_{F}} and a role r_{i} \in \mathbb{R}^{D_{R}} 
        is defined as the tensor product of the filler and role i.e. f_{i} \otimes r_{i} \in \mathbb{R}^{D_{F} \times D_{R}}
        """
        assert len(batched_roles.shape) == 3 and len(batched_fillers.shape) == 3, (f'Expected batched roles' + 
        f'and batched fillers.\n Actual roles: {batched_roles.shape}\tActual fillers: {batched_fillers.shape}\n')
        
        N = batched_roles.shape[0]
        n_roles = self.role_embeddings.weight.shape[0]
        tensor_prod_bindings = torch.einsum('bsf,bsr->bsfr', batched_fillers, batched_roles) # (N_{B}, N_{R}, D_{F}, D_{R})
        
        if self.use_concatenated_rep: 
            z_rep = tensor_prod_bindings.view(N, -1) # (N_{B}, N_{R}*D_{F}*D_{R})
        else: 
            z_rep = tensor_prod_bindings.sum(dim=1).view(N, -1) # (N_{B}, N_{R}, D_{F}, D_{R}) -> (N_{B}, D_{F}, D_{R}) -> (N_{B}, D_{F}*D_{R})
        
        orth_penalty_role = self.get_semi_orth_penalty(self.role_embeddings.weight.t()) # (N_{B}, D_{R}, N_{R})
        role_rank = self.get_rank(self.role_embeddings.weight.t())
        orth_penalty_filler = self.get_semi_orth_penalty(self.filler_embeddings.weight.t())
        filler_rank = self.get_rank(self.filler_embeddings.weight.t())
        
        penalties = torch.stack([orth_penalty_role, orth_penalty_filler], dim=0) 
        coeffs = torch.tensor([self.lambdas_reg[ORTH_PENALTY_ROLE], self.lambdas_reg[ORTH_PENALTY_FILLER]]).to(
            device=self.role_embeddings.weight.device)
        
        encoder_loss = torch.sum(coeffs * penalties)
        
        return {'rep': z_rep, 'bindings': tensor_prod_bindings.reshape(N, n_roles, -1),
                'encoder_logs': {'encoder_loss': encoder_loss, 
                                 ORTH_PENALTY_ROLE: orth_penalty_role, 
                                 ORTH_PENALTY_FILLER: orth_penalty_filler, 
                                 ROLE_RANK: role_rank, 
                                 FILLER_RANK: filler_rank}}
        
    
    @staticmethod
    def get_rank(m: torch.Tensor) -> torch.Tensor: 
        """ Returns rank of matrix """
        return (m.shape[1] - LA.matrix_rank(m))/(m.shape[1] - 1)
    
    @staticmethod
    def get_semi_orth_penalty(m: torch.Tensor, norm_type: str='fro') -> torch.Tensor:
        """ 
        Assumes m is of shape [D, N], i.e. each column corresponds to a vec
        For a semi-orthogonal matrix where D >= N, we have M^{T}M = I_{N}
        """ 
        n = m.shape[-1]
        # ||I - M^{T}M||_{F}^{2}
        mt_m = m.t() @ m
        if norm_type == 'fro': 
            return LA.matrix_norm(torch.eye(n).to(device=mt_m.device) - mt_m, ord='fro')
        else:
            assert norm_type == 'inf', f'Unsupported norm type {norm_type}'
            return LA.matrix_norm(torch.eye(n).to(device=mt_m.device) - mt_m, ord=float('inf'))
        
class TPREncoder(BaseTPREncoder): 
    def __init__(self, n_roles: int, n_fillers: int, role_embed_dim: int,
                 filler_embed_dim: int, use_concatenated_rep: bool,
                 fixed_roles: bool, 
                 lambdas_reg: Dict={ORTH_PENALTY_ROLE: 0, 
                                    ORTH_PENALTY_FILLER: 0},
                init_roles_orth: bool=False,
                init_fillers_orth: bool=False) -> None: 
        super(TPREncoder, self).__init__(n_roles=n_roles, n_fillers=n_fillers, role_embed_dim=role_embed_dim, 
                         filler_embed_dim=filler_embed_dim, use_concatenated_rep=use_concatenated_rep, 
                         lambdas_reg=lambdas_reg, init_roles_orth=init_roles_orth, 
                         init_fillers_orth=init_fillers_orth, fixed_roles=fixed_roles) 
        
    def forward(self, gt_factor_classes: torch.Tensor) -> torch.Tensor: 
        """ 
        return tensor product representation using 
        \sum_{i} f_{i} \otimes r_{i}
        """
        N, _ = gt_factor_classes.shape

        batched_roles = self.role_embeddings.weight.unsqueeze(0).expand(N, -1, -1) # (N, n_roles, embed_dim)
        filler_embed_dim = self.filler_embeddings.weight.shape[1]
        batched_fillers = torch.gather(self.filler_embeddings.weight.unsqueeze(0).expand(N, -1, -1), 
                                       dim=1, index=gt_factor_classes.unsqueeze(-1).expand(-1, -1, filler_embed_dim)) 
        return super().forward(batched_roles=batched_roles, batched_fillers=batched_fillers)      

