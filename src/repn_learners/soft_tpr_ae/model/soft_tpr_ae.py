import torch 
import torch.nn as nn
import torch.linalg as LA 
import torch.nn.functional as F 
from typing import Dict

from src.repn_learners.soft_tpr_ae.model.base_ae import AbstractAE
from src.shared.constants import * 
from src.repn_learners.soft_tpr_ae.optim.loss import bce_recon_loss_fn, mse_recon_loss_fn
from src.repn_learners.soft_tpr_ae.utils import init_embeddings, weights_init
from src.repn_learners.soft_tpr_ae.model.nearest_embed import NearestEmbed
from src.repn_learners.soft_tpr_ae.model.base_tpr_ae import BaseTPREncoder

class Quantiser(nn.Module): 
    """ Quantiser module within the TPR Autoencoder's TPR Decoder (rectangular block in Fig 2 of the 
    main paper). This module contains a learnable filler embedding matrix (self.filler_embeddings), denoted $M_{\psi_{F}}$ in the paper. 
    It quantises each soft filler produced by the Unbinding module into the filler embedding from $M_{\psi_{F}}$ with the smallest
    Euclidean distance.
    
    Inputs: 
        n_fillers (int)                     : number of fillers in $M_{\psi_{F}}$, corresponds to $N_{F}$ in paper
        filler_embed_dim (int)              : dimensionality of filler embedding, corresponds to $D_{F}$ in paper 
        init_embeddings_orth (bool)         : if True, fillers are initialised from semi-orthogonal matrix, 
                                              if False, they are initialised from a normal distribution 
                                              (see init_embeddings in utils.py)
        lambdas_loss (Dict {str: float})    : a dictionary containing the string-based IDs of each loss term 
                                            as keys, and the corresponding coefficient weighting the loss as values 
    """
    def __init__(self, n_fillers: int, filler_embed_dim: int, init_embeddings_orth: bool, 
                 lambdas_loss: Dict={VQ_PENALTY: 1, COMMITMENT_PENALTY: 0.5, 
                                    ORTH_PENALTY_FILLER: 0}) -> None: 
        super().__init__()
        self.filler_embeddings = NearestEmbed(num_embeddings=n_fillers, 
                                       embedding_dim=filler_embed_dim, 
                                       init_orth=init_embeddings_orth)
        self.filler_embed_dim = filler_embed_dim
        self.lambdas_loss = lambdas_loss 

    def make_state(self, soft_fillers: torch.Tensor, 
                   quantised_fillers: torch.Tensor, quantised_fillers_sg: torch.Tensor, 
                   filler_idxs: torch.Tensor) -> Dict: 
        return {SOFT_FILLERS: soft_fillers, 
                QUANTISED_FILLERS: quantised_fillers, 
                QUANTISED_FILLERS_SG: quantised_fillers_sg,
                FILLER_IDXS: filler_idxs}
    
    def forward(self, soft_fillers: torch.Tensor) -> Dict:
        N, n_roles = soft_fillers.shape[:2]
        quantised_fillers_sg, filler_idxs = self.filler_embeddings(soft_fillers, weight_sg=True)
        quantised_fillers, _ = self.filler_embeddings(soft_fillers.detach())

        quantised_fillers_sg = quantised_fillers_sg.view(N, n_roles, self.filler_embed_dim)
        filler_idxs = filler_idxs.view(N, -1)

        quantised_fillers = quantised_fillers.view(N, n_roles, self.filler_embed_dim)
        state = self.make_state(soft_fillers=soft_fillers, quantised_fillers=quantised_fillers,
                                quantised_fillers_sg=quantised_fillers_sg, 
                                filler_idxs=filler_idxs)
        loss = self.get_loss(quantised_fillers=quantised_fillers, soft_fillers=soft_fillers)
        return {'loss': loss, 'state': state}
    
    def get_loss(self, quantised_fillers: torch.Tensor, soft_fillers: torch.Tensor) -> Dict: 
        vq_loss = torch.mean(
            torch.norm((quantised_fillers - soft_fillers.detach())**2, 2, 1)
        )
        commit_loss = torch.mean(
            torch.norm((quantised_fillers.detach() - soft_fillers)**2, 2, 1))

        orth_penalty_filler = BaseTPREncoder.get_semi_orth_penalty(self.filler_embeddings.weight.t())
        penalties = torch.stack([vq_loss, commit_loss, orth_penalty_filler], dim=0)
        lambdas = torch.tensor([self.lambdas_loss[VQ_PENALTY], 
                                self.lambdas_loss[COMMITMENT_PENALTY], 
                                self.lambdas_loss[ORTH_PENALTY_FILLER]]).to(device=penalties.device)
        quantisation_loss = torch.sum(lambdas*penalties)

        filler_rank = BaseTPREncoder.get_rank(self.filler_embeddings.weight)

        return {f'quantiser_{TOTAL_LOSS}': quantisation_loss, 
                ORTH_PENALTY_FILLER: orth_penalty_filler, 
                VQ_PENALTY: vq_loss, 
                COMMITMENT_PENALTY: commit_loss,
                FILLER_RANK: filler_rank}
    
class SoftTPRAutoencoder(AbstractAE): 
    """
    Soft TPR Autoencoder presented in Fig 2 of the main paper. The Soft TPR Autoencoder consists of a standard Encoder, $E$, 
    the TPR Decoder, and a standard decoder, $D$. 
    
    Inputs: 
        encoder (nn.Module)                 : the standard encoder module, corresponding to $E$
        decoder (nn.Module)                 : the standard decoder module, corresponding to $D$
        n_roles (int)                       : the number of roles, corresponding to $N_{R}$
        n_fillers (int)                     : the number of fillers, corresponding to $N_{F}$
        role_embed_dim (int)                : the dimensionality of the role embedding, corresponding to $D_{R}$ 
        filler_embed_dim (int)              : the dimensionality of the filler embedding, corresponding to $D_{F}$
        lambdas_loss(Dict {str: float})     : a dictionary containing the string-based IDs of each loss term 
                                            as keys, and the corresponding coefficient weighting the loss as values 
        init_fillers_orth (bool)            : if True, initialise filler embedding matrix to be a semi-orthogonal matrix
        init_roles_orth (bool)              : as above, but for roles
        freeze_role_embeddings (bool)       : if True, role embeddings are fixed - i.e. they are not updated via backprop
        weakly_supervised (bool)            : if True, apply weak supervision through pair-based method outlined in paper 
        recon_loss_fn (str)                 : specifies the recon loss fn type (either MSE/BCE)
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, 
                 n_roles: int, n_fillers: int, role_embed_dim: int, 
                 filler_embed_dim: int, 
                 lambdas_loss: Dict,
                 init_fillers_orth: bool,
                 init_roles_orth: bool, 
                 freeze_role_embeddings: bool, 
                 weakly_supervised: bool, 
                 recon_loss_fn: str) -> None: 
        super().__init__()
        self.role_embeddings = nn.Embedding(num_embeddings=n_roles, 
                                                 embedding_dim=role_embed_dim).requires_grad_(not freeze_role_embeddings)
        self.quantiser = Quantiser(n_fillers=n_fillers, filler_embed_dim=filler_embed_dim, 
                                   init_embeddings_orth=init_fillers_orth,
                                   lambdas_loss={VQ_PENALTY: lambdas_loss[VQ_PENALTY], 
                                                 COMMITMENT_PENALTY: lambdas_loss[COMMITMENT_PENALTY],
                                                 ORTH_PENALTY_FILLER: lambdas_loss[ORTH_PENALTY_FILLER]})
        
        self.n_roles = n_roles 
        self.n_fillers = n_fillers
        self.filler_embed_dim = filler_embed_dim
        self.role_embed_dim = role_embed_dim
        self.freeze_role_embeddings = freeze_role_embeddings
        self.lambda_recon = lambdas_loss[RECON_PENALTY]
        self.lambda_orth_penalty_role = lambdas_loss[ORTH_PENALTY_ROLE]
        self.lambda_ws_recon = lambdas_loss[WS_RECON_LOSS_PENALTY]
        self.lambda_ws_r_embed_ce = lambdas_loss[WS_DIS_PENALTY]
        self.weakly_supervised = weakly_supervised
        self.recon_loss_fn = recon_loss_fn
        
        self.encoder = encoder 
        self.decoder = decoder 
        self.embed_dim = role_embed_dim * filler_embed_dim

        if freeze_role_embeddings: 
            init_embeddings(init_orth=True, weights=self.role_embeddings.weight)
        else: 
            init_embeddings(init_orth=init_roles_orth, weights=self.role_embeddings.weight)

        weights_init(self.encoder.modules)
        weights_init(self.decoder.modules)

        self.kwargs_for_loading = {
            'n_roles': n_roles, 
            'n_fillers': n_fillers, 
            'role_embed_dim': role_embed_dim, 
            'filler_embed_dim': filler_embed_dim, 
            'lambdas_loss': lambdas_loss, 
            'init_fillers_orth': init_fillers_orth,
            'init_roles_orth': init_roles_orth,
            'freeze_role_embeddings': freeze_role_embeddings, 
            'weakly_supervised': weakly_supervised,
            'recon_loss_fn': recon_loss_fn
        }

    def decode(self, x: torch.Tensor) -> torch.Tensor: 
        return self.decoder(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor: 
        return self.encoder(x)
    
    def repn_fn(self, x: torch.Tensor, key: str=QUANTISED_FILLERS) -> torch.Tensor: 
        """ 
        The Soft TPR corresponds to the encoder's output. 
        This representation function considers alternative representational forms.
        Inputs: 
            x (torch.Tensor)            : input image 
            key (str)                   : if QUANTISED_FILLERS              -> for each data point, $x_{i}$, return a $(N_{R}, D_{F})$-sized matrix 
                                                                            corresponding to the quantised fillers of $x_{i}$
                                          if QUANTISED_FILLERS_CONCATENATED -> concatenate each filler embedding to produce a single 
                                                                            $N_{R}*D_{F}$-dimensional vector
                                          if SOFT_FILLERS                   -> for each $x_{i}$, return a $(N_{R}, D_{F})$-sized matrix 
                                                                            corresponding to the *soft* fillers of $x_{i}$
                                          if SOFT_FILLERS_CONCATENATED      -> the same as QUANTISED_FILLERS_CONCATENATED, but SOFT_FILLERS are used
                                          if FILLER_IDXS                    -> for each $x_{i}, return a list of IDs for each quantised filler. These numerical
                                                                            IDs are each quantised filler's column number in the Quantiser's filler_embedding matrix
                                          if Z_TPR                          -> return the (traditional) TPR corresponding to the Encoder output's explicit TPR counterpart
                                                                            with the greedily defined form of Eq 5 (main paper)
                                          if TPR_BINDINGS                   -> for each $x_{i}$, return a $(N_{R}, D_{F}, D_{R})$-dimensional tensor corresponding
                                                                            to that $x_{i}$'s role-filler binding embeddings i.e., $\psi_{F}(f_{m(i)}) \otimes \psi_{R}(r_{i})$ 
                                                                            as defined in the paper. 
                                          if TPR_BINDINGS_FLATTENED         -> flatten TPR_BINDINGS to produce a $(N_{R}*D_{F}*D_{R})$-dimensional vector
                                          if Z_SOFT_TPR                     -> return the soft TPR for each $x_{i}$, i.e. the output of the Encoder. 
                                                                            Note that the encoder output will have dimension $(self.filler_embed_dim, self.role_embed_dim)$.
                                                                            This is flattened to be a vector (isomorphism between $\mathbb{R}^{D_{F} \cdot D_{R}}$ 
                                                                            and $\mathbb{R}^{D_{F}} \otimes \mathbb{R}^{D_{R}}$)
        Returns: 
            torch.Tensor                : tensor that will be used as representation vector for subsequent downstream tasks
        """
        z = self.encode(x) 
        soft_fillers = self.unbind(z)
        quantised_out = self.quantiser(soft_fillers)['state'] 
        if SOFT_FILLERS in key or QUANTISED_FILLERS in key:
            if SOFT_FILLERS in key: 
                fillers = quantised_out[SOFT_FILLERS]                                           # (N, N_{R}, D_{F})
            elif QUANTISED_FILLERS in key: 
                fillers = quantised_out[QUANTISED_FILLERS_SG]
            if CONCATENATED in key: 
                fillers = fillers.view(x.shape[0], -1)                                          # (N, N_{R}, D_{F}) -> (N, N_{R}*D_{F})
            return fillers
        if key == FILLER_IDXS: 
            return quantised_out[FILLER_IDXS].to(torch.float32)                                 # (N, N_{R})
        if key == Z_SOFT_TPR: 
            return z.view(-1, self.embed_dim)                                                   # soft approximation to a TPR (N, D_{F}*D_{R})
        if key == Z_TPR or TPR_BINDINGS in key: 
            constructor_out = self.construct(quantised_out[QUANTISED_FILLERS_SG])
            if key == Z_TPR: 
                return constructor_out[Z_TPR]                                                   # (N, D_{F}*D_{R})
            else: 
                bindings = constructor_out[TPR_BINDINGS].view(x.shape[0], self.n_roles, -1)     # (N, N_{R}, D_{F}, D_{R}) -> (N, N_{R}, D_{F}*D_{R})
                if CONCATENATED in key: 
                    bindings = bindings.view(x.shape[0], -1)                                    # (N, N_{R}, D_{F}*D_{R})
                return bindings
    
    def make_state(self, x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor, 
                   z_tpr: torch.Tensor, quantised_fillers: torch.Tensor, soft_fillers: torch.Tensor, 
                   filler_idxs: torch.Tensor) -> Dict: 
        return {'x': x, 
                'x_hat': x_hat, 
                Z_SOFT_TPR: z, 
                Z_TPR: z_tpr, 
                QUANTISED_FILLERS: quantised_fillers, 
                SOFT_FILLERS: soft_fillers,
                FILLER_IDXS: filler_idxs}
        
    def unbind(self, z: torch.Tensor) -> torch.Tensor:
        """ 
        Performs unbinding operation 
        Note that as per the 'Unbinding' section of B.3.1. If the role embedding matrix, $M_{\psi_{R}}$ is semi-orthogonal,
        then, it is left-invertible, and the left inverse is the transpose of $M_{\psi_{R}}$. Taking the inner product
        between the left inverse (transpose) of $M_{\psi_{R}}$ and $z$ recovers the soft fillers, as per Eqs 3-4 in the main paper. 
        
        Inputs: 
            z (torch.Tensor)                        : soft TPR of dimension (N, D_{F}, D_{R})
        Returns: 
            torch.Tensor                            : tensor of dimension (N, N_{R}, D_{F}) containing the unbound soft fillers
                                                      for each data point
        """ 
        soft_fillers = torch.bmm(z, self.role_embeddings.weight.unsqueeze(0).expand(z.shape[0], -1, -1).permute(0, 2, 1))       # (N, D_{F}, D_{R}) @ (N, N_{R}, D_{R}) -> (N, D_{F}, N_{R})
        soft_fillers = soft_fillers.permute(0, 2, 1).contiguous()                                                               # -> (N, N_{R}, D_{F})
        return soft_fillers
    
    def construct(self, quantised_fillers: torch.Tensor) -> Dict: 
        """ 
        Apply TPR construction operation to the quantised fillers and the role embeddings 
        to retrieve an explicit TPR. Also returns the TPR bindings 
        
        Inputs: 
            quantised_fillers (torch.Tensor): a tensor of dimension (N, N_{R}, D_{F}) where 
                quantised_fillers[:,i] contains the $i$-th role's corresponding $D_{F}$-dimensional bound filler
                for each data point
        
        Returns: 
            Dict {str: torch.Tensor}: a dictionary containing:
                the explicit TPR, a tensor of dimension (N, D_{F}*D_{R}),
                the TPR bindings, a tensor of dimension (N, N_{R}, D_{F}, D_{R})
        """
        batched_roles = self.role_embeddings.weight.unsqueeze(0).expand(quantised_fillers.shape[0], -1, -1)
        tpr_bindings = torch.einsum('bsf,bsr->bsfr', quantised_fillers, batched_roles)
        z_tpr = tpr_bindings.sum(dim=1).view(-1, self.embed_dim)
        return {Z_TPR: z_tpr,
                TPR_BINDINGS: tpr_bindings}
    
    def get_ws_out(self, quantised_out: Dict, 
                        x: torch.Tensor, gt_factor_classes: torch.Tensor) -> torch.Tensor: 
        """ 
        Computes all outputs associated with weak supervision as explained in the main paper. 
        Computes the weakly supervised reconstruction loss term and the weakly supervised cross entropy term corresponding to the last 2 terms in Eq 7
        (Recall that each pair of images, $(x, x')$ differ in the FoV *value* for only 1 FoV type. I.e., there is 1 difference
        between the pair of images)
        
        Inputs:
            quantised_fillers_sg (torch.Tensor)     : tensor of dimension (N, N_{R}, D_{F}) representing quantised fillers (note that the quantised fillers themselves are 
                                                    FIXED in all other operations other than the VQ-VAE quantisation-based loss applied by the quantiser module)
            tpr_bindings (torch.Tensor)             : tensor of dimension (N, N_{R}, D_{F}*D_{R}) representing the role-filler embedding bindings for each data point
            x (torch.Tensor)                        : tensor of dimension (N, C, H, W)
            gt_factor_classes (torch.Tensor)        : tensor of dimension (N, gt_N_{R}) where gt_factor_classes[i, k] contains an integer ID denoting the identity of the 
                                                    $i$-th datapoint's filler bound to role $k$, and gt_N_{R} denotes the GT number of roles in the dataset
        """ 
        
        recon_loss = 0 
        ws_ce_loss = 0 

        quantised_fillers_sg = quantised_out[QUANTISED_FILLERS_SG]                                                      # (N, N_{R}, D_{F}) (note that N = 2*N_{B}, where N_{B} denotes the batch-size, as inputs are paired)
        tpr_bindings = self.construct(quantised_fillers_sg)[TPR_BINDINGS].detach() 

        quantised_fillers_sg = torch.stack(torch.chunk(quantised_fillers_sg, 2, 0), dim=1)                              # (N_{B}, 2, N_{R}, D_{F})
        N = quantised_fillers_sg.shape[0]
        dist = LA.vector_norm(quantised_fillers_sg[:, 0] - quantised_fillers_sg[:, 1], 2, dim=-1) + 1e-8                # (N_{B}, N_{R}, D_{F}) -> (N_{B}, N_{R}, 1)
        gt1, gt2 = torch.chunk(gt_factor_classes, 2, 0)
        one_hot = (gt1 != gt2).to(torch.float16) # (N_{B}, N_{R}) 
        
        if one_hot.shape != dist.shape:                                                                                 # (gt_N_{R}) != (N_{R'})
            diff = dist.shape[1] - one_hot.shape[1]
            one_hot = torch.concatenate([one_hot, torch.zeros(size=(one_hot.shape[0], diff)).cuda()], dim=-1)
            
        ws_ce_loss = F.cross_entropy(dist, one_hot)
        
        mask = F.gumbel_softmax(dist, dim=1, hard=True).unsqueeze(-1).expand(-1, -1, self.filler_embed_dim).to(bool)    # (N_{B}, N_{R})
        tpr_bindings_split = torch.stack(torch.chunk(tpr_bindings, 2, 0), dim=1)                                        # (N_{B}, 2, N_{R}, D_{F}, D_{R})
        
        mask_for_tpr = mask.unsqueeze(-1).expand(-1, -1, -1, self.role_embed_dim)
        temp0 = tpr_bindings_split[:, 0][mask_for_tpr].reshape(N, self.filler_embed_dim, self.role_embed_dim) 
        temp1 = tpr_bindings_split[:, 1][mask_for_tpr].reshape(N, self.filler_embed_dim, self.role_embed_dim)
        
        tpr_bindings_split[:, 0][mask_for_tpr] = temp1.view(-1)
        tpr_bindings_split[:, 1][mask_for_tpr] = temp0.view(-1)
        swapped_bindings = tpr_bindings_split 
        tpr_bindings_split = torch.concatenate(torch.unbind(tpr_bindings_split, dim=1), dim=0)                          # (2*N_{B}, N_{R}, D_{F}, D_{R})
        swapped_tpr = tpr_bindings_split.sum(dim=1).view(-1, self.embed_dim)

        x_hat = self.decode(swapped_tpr)
        x1, x2 = torch.chunk(x, 2, dim=0)
        
        with torch.no_grad():
            mse_recon_loss = mse_recon_loss_fn(x_hat, torch.concatenate([x2, x1], dim=0), logging=True).detach()                 # for logging purposes
            bce_recon_loss = bce_recon_loss_fn(x_hat, torch.concatenate([x2, x1], dim=0), logging=True).detach()
        
        if self.recon_loss_fn == 'mse': 
            recon_loss = mse_recon_loss_fn(x_hat, torch.concatenate([x2, x1], dim=0), logging=False)
        else: 
            recon_loss = bce_recon_loss_fn(x_hat, torch.concatenate([x2, x1], dim=0), logging=False)
        
        return {'loss': {f'{WEAKLY_SUPERVISED}_{MSE_RECON_LOSS}': mse_recon_loss, 
                         f'{WEAKLY_SUPERVISED}_{BCE_RECON_LOSS}': bce_recon_loss,
                         f'{WEAKLY_SUPERVISED}_{RECON_LOSS}': recon_loss,
                         f'{WEAKLY_SUPERVISED}_{CE_LOSS}': ws_ce_loss}, 
                'state': {'swapped_tpr': swapped_tpr, 
                f'{WEAKLY_SUPERVISED}_argmax': torch.argwhere(mask == 1)[:, 1], 
                'swapped_bindings': swapped_bindings,
                f'{WEAKLY_SUPERVISED}_x_hat': x_hat}}  
        
    def forward(self, x: torch.Tensor, gt_factor_classes: torch.Tensor) -> Dict: 

        z = self.encode(x)                                                                                              # (N, D_{F}, D_{R})
        soft_fillers = self.unbind(z) 
        quantiser_out = self.quantiser(soft_fillers)
        quantised_out, quantiser_loss = quantiser_out['state'], quantiser_out['loss']
        
        quantised_fillers_sg, filler_idxs = quantised_out[QUANTISED_FILLERS_SG], quantised_out[FILLER_IDXS]
        construct_output = self.construct(quantised_fillers_sg)
        z_tpr = construct_output[Z_TPR]
        
        x_hat = self.decode(z_tpr)
        state = self.make_state(x=x, x_hat=x_hat, z=z, z_tpr=z_tpr, 
                                quantised_fillers=quantised_fillers_sg, 
                                soft_fillers=soft_fillers, 
                                filler_idxs=filler_idxs)
        loss = self.get_us_loss(x=x, x_hat=x_hat, quantiser_loss=quantiser_loss)
        loss[TOTAL_LOSS] = loss[f'us_{TOTAL_LOSS}']
        
        if self.weakly_supervised and self.train: 
            ws_out = self.get_ws_out(quantised_out=quantised_out,
                                                   x=x, gt_factor_classes=gt_factor_classes)
            loss[TOTAL_LOSS] += (self.lambda_ws_recon*ws_out['loss'][f'ws_{RECON_LOSS}'] + 
                                   self.lambda_ws_r_embed_ce*ws_out['loss'][f'ws_{CE_LOSS}']
                                   ) 
            return {'loss': {**loss, **ws_out['loss']}, 'state': {**state, **ws_out['state']}}
        
        return {'loss': loss, 'state': state}

    def get_us_loss(self, x: torch.Tensor, x_hat: torch.Tensor, quantiser_loss: Dict) -> torch.Tensor: 
        loss_logs = {}

        with torch.no_grad():
            mse_recon_loss = mse_recon_loss_fn(x_hat, x, logging=True).detach()                                              # logging
            bce_recon_loss = bce_recon_loss_fn(x_hat, x, logging=True).detach()
        if self.recon_loss_fn == MSE: 
            recon_loss = mse_recon_loss_fn(x_hat, x, logging=False)
        else: 
            recon_loss = bce_recon_loss_fn(x_hat, x, logging=False)
        
        loss_logs[RECON_LOSS] = recon_loss
        total_loss = self.lambda_recon*recon_loss + quantiser_loss[f'quantiser_{TOTAL_LOSS}']
        
        if self.lambda_orth_penalty_role != 0 and not self.freeze_role_embeddings: 
            orth_penalty_role_loss = BaseTPREncoder.get_semi_orth_penalty(self.role_embeddings.weight.t()) * self.lambda_orth_penalty_role
            total_loss += orth_penalty_role_loss
            loss_logs = {ORTH_PENALTY_ROLE: orth_penalty_role_loss}

        role_rank = BaseTPREncoder.get_rank(self.role_embeddings.weight.t())

        return {f'{UNSUPERVISED}_{MSE_RECON_LOSS}': mse_recon_loss,
                f'{UNSUPERVISED}_{BCE_RECON_LOSS}': bce_recon_loss,
                f'{UNSUPERVISED}_{TOTAL_LOSS}': total_loss, 
                ROLE_RANK: role_rank,
                **quantiser_loss,
                **loss_logs}

    def count_params(self): 
        return sum(p.numel() for p in self.encoder.parameters()) + sum(p.numel() for p in self.decoder.parameters()) + sum(p.numel() for p in self.quantiser.parameters())