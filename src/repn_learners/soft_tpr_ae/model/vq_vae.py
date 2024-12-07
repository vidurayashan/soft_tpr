import torch 
import torch.nn as nn
import torch.linalg as LA 
import torch.nn.functional as F 
import numpy as np
from typing import Dict 

from src.shared.constants import * 
from src.repn_learners.soft_tpr_ae.optim.loss import mse_recon_loss_fn, bce_recon_loss_fn

from torch.autograd import Function 

class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)
                                      ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

class VQVAE(nn.Module):
    def __init__(self, d, k=10, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3, **kwargs):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                d, num_channels, kernel_size=4, stride=2, padding=1),
        )
        
        self.d = d
        self.emb = NearestEmbed(k, d)
        self.latent_dim = d*256
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.nc = num_channels
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)
        
        self.kwargs_for_loading = {
            'd': d, 
            'k': k, 
            'bn': bn,
            'vq_coef': vq_coef, 
            'commt_coef': commit_coef, 
            'num_channels': num_channels,
            **kwargs
        }

    def encode(self, x):
        return self.encoder(x) # (N_{B}, D, H, W)

    def decode(self, x):
        return torch.tanh(self.decoder(x))
    
    def repn_fn(self, x: torch.Tensor): 
        # encoder produces shape (N_{B}, D, H, W)
        # -> (N_{B}, H*W, D) where we have H*W fillers, each of dimension D
        state = self(x, None)['state']
        quantised_embeddings = state['quantised_embeddings'] # (N_{B}, H*W, D)
        return quantised_embeddings.contiguous().reshape(x.shape[0], -1)
        #return self.encode(x).permute(0, 2, 3, 1).contiguous().reshape(x.shape[0], -1)
    
    def make_state(self, x: torch.Tensor, x_hat: torch.Tensor, 
                   approx_embeddings: torch.Tensor, 
                   quantised_embeddings: torch.Tensor, embedding_idxs: torch.Tensor) -> Dict: 
        return {'x': x, 
                'x_hat': x_hat, 
                'approx_embeddings': approx_embeddings, 
                'quantised_embeddings': quantised_embeddings, 
                'embedding_idxs': embedding_idxs}

    def forward(self, x, gt_factor_classes) -> Dict:
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        x_hat = self.decode(z_q)
        state = self.make_state(x=x, x_hat=x_hat, 
                                approx_embeddings=z_e, 
                                quantised_embeddings=z_q, 
                                embedding_idxs=argmin)
        loss_out = self.loss_function(x=x, recon_x=x_hat, z_e=z_e, emb=emb)
        return {'loss': loss_out, 'state': state}

    def sample(self, size):
        sample = torch.randn(size, self.d, self.f,
                             self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.d, self.f, self.f)).cpu()

    def loss_function(self, x, recon_x, z_e, emb) -> Dict:
        self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
        self.commit_loss = torch.mean(
            torch.norm((emb.detach() - z_e)**2, 2, 1))

        total_loss = self.mse + self.vq_coef*self.vq_loss + self.commit_coef*self.commit_loss
        return {'total_loss': total_loss, 
                'recon_loss': self.mse, 
                'mse_recon_loss': mse_recon_loss_fn(recon_x, x, logging=True, postprocessing=lambda x: x),
                'bce_recon_loss': bce_recon_loss_fn(recon_x.mul(0.5).add(0.5), x.mul(0.5).add(0.5), logging=True, logits=False),
                VQ_PENALTY: self.vq_loss, 
                COMMITMENT_PENALTY: self.commit_loss}
    
    def count_params(self): 
        return sum(p.numel() for p in self.encoder.parameters()) + sum(p.numel() for p in self.decoder.parameters()) + sum(p.numel() for p in self.emb.parameters())
