"""
This code has been adapted from the supplementary material linked to the paper 
'Visual Representation Learning Does Not Generalize Strongly Within the Same Domain' (ICLR 2022) 
accessible at https://openreview.net/forum?id=9RUHPlladgh

@inproceedings{
schott2022visual,
title={Visual Representation Learning Does Not Generalize Strongly Within the Same Domain},
author={Lukas Schott and Julius Von K{\"u}gelgen and Frederik Tr{\"a}uble and Peter Vincent Gehler and Chris Russell and Matthias Bethge and Bernhard Sch{\"o}lkopf and Francesco Locatello and Wieland Brendel},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=9RUHPlladgh}
}
"""

import numpy as np
import torch
import torch.nn.init as init
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Encoder(nn.Sequential):
    def __init__(self, number_channels: int = 3,
                 number_latents: int = 10):
        self.number_channels = number_channels
        self.number_latents = number_latents
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(number_channels, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256 * 1 * 1)),  # B, 256
            nn.Linear(256, number_latents * 2),  # B, z_dim*2
        )


class Decoder(nn.Sequential):
    def __init__(self, number_channels: int = 3,
                 number_latents: int = 10):
        super().__init__()
        self.number_channels = number_channels
        self.number_latents = number_latents
        self.net = nn.Sequential(
            nn.Linear(number_latents, 256),  # B, 256
            View((-1, 256, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, number_channels, 4, 2, 1),  # B, nc, 64, 64
        )


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


class BetaVAE(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, number_latents: int = 10, number_channels: int = 3,
                 beta=1, decoder_dist='bernoulli'):
        super().__init__()
        self.number_latents = number_latents
        self.decoder_dist = decoder_dist
        self.beta = beta

        self.encoder = Encoder(number_channels=number_channels,
                               number_latents=number_latents)
        self.decoder = Decoder(number_channels=number_channels,
                               number_latents=number_latents)
        self.weight_init()
        self.kwargs_for_loading = {
            'number_latents': number_latents, 
            'number_channels': number_channels,
            'beta': beta, 
            'decoder_dist': decoder_dist
        }

    def weight_init(self):
        for net in self._modules:
            for block in self._modules[net]:
                for m in block._modules.values():
                    kaiming_init(m)
                    
    def repn_fn(self, x, key=None): 
        return self(x, use_stochastic=False)

    def forward(self, x, use_stochastic=False):
        distributions = self.encoder(x)
        mu = distributions[:, :self.number_latents]
        logvar = distributions[:, self.number_latents:]
        if not use_stochastic:
            return mu

        z = reparametrize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def loss_f(self, x, x_recon, mu, logvar, reduce=True):
        mse_recon_loss = reconstruction_loss(x, x_recon,
                                         'gaussian',
                                         x.shape[0],
                                         reduction='sum' if reduce else 'none')
        bce_recon_loss = reconstruction_loss(x, x_recon,
                                         'bernoulli',
                                         x.shape[0],
                                         reduction='sum' if reduce else 'none')
        if self.decoder_dist == 'bernoulli': 
            recon_loss = bce_recon_loss
        else: 
            recon_loss = mse_recon_loss
            
        normal_entropy = compute_ent_normal(logvar)
        cross_ent_normal = compute_cross_ent_normal(mu, logvar)
        # sum over latent dimensions, mean over batch dimension
        kl_normal = (cross_ent_normal - normal_entropy).sum(dim=1).mean(dim=0)
        kl_per_dim = (cross_ent_normal - normal_entropy).mean(dim=0)
        vae_loss = recon_loss + self.beta * kl_normal
        infos = {'kl_normal': kl_normal.item(), 
                 'total_loss': vae_loss.item(),
                 'mse_recon_loss': mse_recon_loss.item(),
                 'bce_recon_loss': bce_recon_loss.item(),
                 'kl_per_dim': kl_per_dim}
        return vae_loss, infos


class SlowVAE(BetaVAE):
    def __init__(self, number_latents: int = 10, number_channels: int = 3,
                 beta=1, gamma=1, rate_prior=1,
                 decoder_dist='bernoulli'):
        super().__init__(number_latents=number_latents,
                         number_channels=number_channels, beta=beta,
                         decoder_dist=decoder_dist)
        self.gamma = gamma
        self.rate_prior = \
            torch.nn.Parameter(torch.tensor([rate_prior]).float(),
                               requires_grad=False)

        self.normal_mean = torch.nn.Parameter(torch.zeros(self.number_latents),
                                              requires_grad=False)
        self.normal_sigma = torch.nn.Parameter(torch.ones(self.number_latents),
                                               requires_grad=False)
        self.normal_dist = torch.distributions.normal.Normal(
            self.normal_mean, self.normal_sigma)
        self.kwargs_for_loading = {
            **self.kwargs_for_loading, 
            'beta': beta, 
            'gamma': gamma, 
            'rate_prior': rate_prior
        }

    def compute_cross_ent_laplace(self, mean, logvar, rate_prior):
        var = torch.exp(logvar)
        sigma = torch.sqrt(var)
        ce = - torch.log(rate_prior / 2) + rate_prior * sigma * \
             np.sqrt(2 / np.pi) * torch.exp(- mean ** 2 / (2 * var)) - \
             rate_prior * mean * (
                     1 - 2 * self.normal_dist.cdf(mean / sigma))
        return ce

    def compute_cross_ent_combined(self, mu, logvar):
        normal_entropy = compute_ent_normal(logvar)
        cross_ent_normal = compute_cross_ent_normal(mu, logvar)
        # assuming couples, do Laplace both ways
        mu0 = mu[::2]
        mu1 = mu[1::2]
        logvar0 = logvar[::2]
        logvar1 = logvar[1::2]
        rate_prior0 = self.rate_prior
        rate_prior1 = self.rate_prior
        cross_ent_laplace = (
                self.compute_cross_ent_laplace(mu0 - mu1, logvar0,
                                               rate_prior0) +
                self.compute_cross_ent_laplace(mu1 - mu0, logvar1, rate_prior1))
        return ([x.sum(1).mean(0, keepdim=True)
                for x in
                [normal_entropy, cross_ent_normal, cross_ent_laplace]], 
        [x.mean(0) for x in [normal_entropy, cross_ent_normal, cross_ent_laplace]])

    def loss_f(self, x, x_recon, mu, logvar, use_stochastic=False):
        mse_recon_loss = reconstruction_loss(x, x_recon,
                                         'gaussian',
                                         x.shape[0],
                                         reduction='sum')
        bce_recon_loss = reconstruction_loss(x, x_recon,
                                         'bernoulli',
                                         x.shape[0],
                                         reduction='sum')
        if self.decoder_dist == 'bernoulli': 
            recon_loss = bce_recon_loss
        else: 
            recon_loss = mse_recon_loss
        # train both ways
        acc_stats, per_dim_stats = self.compute_cross_ent_combined(mu, logvar)
        normal_entropy, cross_ent_normal, cross_ent_laplace = acc_stats 
        normal_entropy_per_dim, cross_ent_normal_per_dim, cross_ent_laplace_per_dim = per_dim_stats 

        vae_loss = 2 * recon_loss
        kl_normal = cross_ent_normal - normal_entropy
        kl_laplace = cross_ent_laplace - normal_entropy

        kl_normal_per_dim = cross_ent_normal_per_dim - normal_entropy_per_dim
        kl_laplace_per_dim = cross_ent_laplace_per_dim - normal_entropy_per_dim 

        vae_loss = vae_loss + self.beta * kl_normal
        vae_loss = vae_loss + self.gamma * kl_laplace
        infos = {'kl_normal': kl_normal.item(), 'total_loss': vae_loss.item(),
                 'mse_recon_loss': mse_recon_loss.item(),
                 'bce_recon_loss': bce_recon_loss.item(),
                 'kl_laplace': kl_laplace.item(), 
                 'kl_per_dim': self.beta*kl_normal_per_dim + self.gamma*kl_laplace_per_dim}
        return vae_loss, infos

class MLVAEModel(BetaVAE): 
    def repn_fn(self, x: torch.Tensor, key=None): 
        return self(x, gt_factor_classes=None, use_stochastic=False)
    
    def forward(self, x: torch.Tensor, gt_factor_classes: torch.Tensor, use_stochastic: bool=False): 
        distributions = self.encoder(x) 
        mu = distributions[:, :self.number_latents]
        if not use_stochastic: 
            return mu 
        
        logvar = distributions[:, self.number_latents:]
        bs = x.shape[0]
        assert bs % 2 == 0 
        # t, t-1 stored as interleafed pairs in batch dim 
        mu0 = mu[::2]
        mu1 = mu[1::2]
        logvar0 = logvar[::2]
        logvar1 = logvar[1::2]

        var_0 = logvar0.exp()
        var_1 = logvar1.exp()
        # a implemented by taking product of encoder distributions as in GVAE by Hosoya
        new_var = 2*var_0*var_1 / (var_0 + var_1)
        new_mu = (mu0/var_0 + mu1/var_1)*new_var * 0.5
        new_log_var = torch.log(new_var)

        gt_factor_classes0 = gt_factor_classes[::2]
        gt_factor_classes1 = gt_factor_classes[1::2]
        s_mask = (gt_factor_classes0 != gt_factor_classes1)
        zeros = torch.zeros(size=(s_mask.shape[0], self.number_latents-s_mask.shape[1])).cuda() 
        s_mask = torch.concatenate([s_mask, zeros], dim=1).to(bool)

        mean_sample_0, log_var_sample_0 = aggregate_s_known(
            mu0, logvar0, new_mu, new_log_var, s_mask) # replace masked dimensions with averaged statistic
        mean_sample_1, log_var_sample_1 = aggregate_s_known(
            mu1, logvar1, new_mu, new_log_var, s_mask)

        z_sampled_0 = reparametrize(mean_sample_0, log_var_sample_0)
        z_sampled_1 = reparametrize(mean_sample_1, log_var_sample_1)

        # stack alternating on batch dimension [z0_t0, z0_t1, z1_t0, z1_t1, ...]
        interleaf_stacked_z = torch.stack([z_sampled_0, z_sampled_1],
                                          dim=1).flatten(0, 1)
        x_recon = self.decoder(interleaf_stacked_z)
        
        
        mse_recon_loss = reconstruction_loss(x, x_recon,
                                         'gaussian',
                                         x.shape[0],
                                         reduction='sum')
        bce_recon_loss = reconstruction_loss(x, x_recon,
                                         'bernoulli',
                                         x.shape[0],
                                         reduction='sum')
        if self.decoder_dist == 'bernoulli': 
            recon_loss = bce_recon_loss
        else: 
            recon_loss = mse_recon_loss

        # compute KL with standard normal prior
        kl_loss_0 = compute_kl(mean_sample_0, torch.zeros_like(mean_sample_0),
                               log_var_sample_0,
                               torch.zeros_like(log_var_sample_0))
        kl_loss_1 = compute_kl(mean_sample_1, torch.zeros_like(mean_sample_1),
                               log_var_sample_1,
                               torch.zeros_like(log_var_sample_1))
        # sum over latent dimensions, mean over batch dimension
        kl_loss = torch.cat([kl_loss_0 + kl_loss_1], dim=0).sum(dim=1).mean()
        kl_per_dim = torch.cat([kl_loss_0 + kl_loss_1], dim=0).mean(dim=0)
        vae_loss = self.beta * kl_loss + recon_loss
        infos = {'kl_normal_gvae': kl_loss.item(),
                 'total_loss': vae_loss.item(),
                 'kl_per_dim': kl_per_dim,
                 'mse_recon_loss': mse_recon_loss.item(),
                 'bce_recon_loss': bce_recon_loss.item()}
        return vae_loss, infos, x_recon

class GVAEModel(BetaVAE): 
    def repn_fn(self, x: torch.Tensor, key=None): 
        return self(x, gt_factor_classes=None, use_stochastic=False) 
    
    def forward(self, x: torch.Tensor, gt_factor_classes: torch.Tensor, use_stochastic: bool=False): 
        distributions = self.encoder(x) 
        mu = distributions[:, :self.number_latents]
        if not use_stochastic: 
            return mu 
        
        logvar = distributions[:, self.number_latents:]
        bs = x.shape[0]
        assert bs % 2 == 0 
        # t, t-1 stored as interleafed pairs in batch dim 
        mu0 = mu[::2]
        mu1 = mu[1::2]
        logvar0 = logvar[::2]
        logvar1 = logvar[1::2]

        var_0 = logvar0.exp()
        var_1 = logvar1.exp()
        # a implemented by averaging as in Bouchacourt et al (2018) paper 
        new_log_var = (0.5 * var_0 + 0.5 * var_1).log()
        new_mu = 0.5 * mu0 + 0.5 * mu1

        gt_factor_classes0 = gt_factor_classes[::2]
        gt_factor_classes1 = gt_factor_classes[1::2]
        s_mask = (gt_factor_classes0 != gt_factor_classes1)
        zeros = torch.zeros(size=(s_mask.shape[0], self.number_latents-s_mask.shape[1])).cuda() 
        s_mask = torch.concatenate([s_mask, zeros], dim=1).to(bool)


        mean_sample_0, log_var_sample_0 = aggregate_s_known(
            mu0, logvar0, new_mu, new_log_var, s_mask) # replace masked dimensions with averaged statistic
        mean_sample_1, log_var_sample_1 = aggregate_s_known(
            mu1, logvar1, new_mu, new_log_var, s_mask)

        z_sampled_0 = reparametrize(mean_sample_0, log_var_sample_0)
        z_sampled_1 = reparametrize(mean_sample_1, log_var_sample_1)

        # stack alternating on batch dimension [z0_t0, z0_t1, z1_t0, z1_t1, ...]
        interleaf_stacked_z = torch.stack([z_sampled_0, z_sampled_1],
                                          dim=1).flatten(0, 1)
        x_recon = self.decoder(interleaf_stacked_z)
        
        
        mse_recon_loss = reconstruction_loss(x, x_recon,
                                         'gaussian',
                                         x.shape[0],
                                         reduction='sum')
        bce_recon_loss = reconstruction_loss(x, x_recon,
                                         'bernoulli',
                                         x.shape[0],
                                         reduction='sum')
        if self.decoder_dist == 'bernoulli': 
            recon_loss = bce_recon_loss
        else: 
            recon_loss = mse_recon_loss

        # compute KL with standard normal prior
        kl_loss_0 = compute_kl(mean_sample_0, torch.zeros_like(mean_sample_0),
                               log_var_sample_0,
                               torch.zeros_like(log_var_sample_0))
        kl_loss_1 = compute_kl(mean_sample_1, torch.zeros_like(mean_sample_1),
                               log_var_sample_1,
                               torch.zeros_like(log_var_sample_1))
        # sum over latent dimensions, mean over batch dimension
        kl_loss = torch.cat([kl_loss_0 + kl_loss_1], dim=0).sum(dim=1).mean()
        kl_per_dim = torch.cat([kl_loss_0 + kl_loss_1], dim=0).mean(dim=0)
        vae_loss = self.beta * kl_loss + recon_loss
        infos = {'kl_normal_gvae': kl_loss.item(),
                 'total_loss': vae_loss.item(),
                 'kl_per_dim': kl_per_dim,
                 'mse_recon_loss': mse_recon_loss.item(),
                 'bce_recon_loss': bce_recon_loss.item()}
        return vae_loss, infos, x_recon

class AdaGVAE(BetaVAE):
    """
    Pytorch implementation of the Ada-GVAE model from
    https://arxiv.org/pdf/2002.02886.pdf.
    Implementation inspired by
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/weak/weak_vae.py#L62
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/weak/weak_vae.py#L317
    """
    def forward(self, x: torch.tensor, use_stochastic=False):
        distributions = self.encoder(x)
        mu = distributions[:, :self.number_latents]
        if not use_stochastic:
            return mu

        logvar = distributions[:, self.number_latents:]
        bs = x.shape[0]
        assert bs % 2 == 0
        # t, t-1 stored as interleafed pairs in batch dimension
        mu0 = mu[::2]
        mu1 = mu[1::2]
        logvar0 = logvar[::2]
        logvar1 = logvar[1::2]
        kl_per_point = compute_kl(mu0, mu1, logvar0, logvar1)

        var_0 = logvar0.exp()
        var_1 = logvar1.exp()
        new_log_var = (0.5 * var_0 + 0.5 * var_1).log()
        new_mu = 0.5 * mu0 + 0.5 * mu1

        mean_sample_0, log_var_sample_0 = aggregate_argmax(
            mu0, logvar0, new_mu, new_log_var, kl_per_point)
        mean_sample_1, log_var_sample_1 = aggregate_argmax(
            mu1, logvar1, new_mu, new_log_var, kl_per_point)

        z_sampled_0 = reparametrize(mean_sample_0, log_var_sample_0)
        z_sampled_1 = reparametrize(mean_sample_1, log_var_sample_1)

        # stack alternating on batch dimension [z0_t0, z0_t1, z1_t0, z1_t1, ...]
        interleaf_stacked_z = torch.stack([z_sampled_0, z_sampled_1],
                                          dim=1).flatten(0, 1)
        x_recon = self.decoder(interleaf_stacked_z)
        
        
        mse_recon_loss = reconstruction_loss(x, x_recon,
                                         'gaussian',
                                         x.shape[0],
                                         reduction='sum')
        bce_recon_loss = reconstruction_loss(x, x_recon,
                                         'bernoulli',
                                         x.shape[0],
                                         reduction='sum')
        if self.decoder_dist == 'bernoulli': 
            recon_loss = bce_recon_loss
        else: 
            recon_loss = mse_recon_loss

        # compute KL with standard normal prior
        kl_loss_0 = compute_kl(mean_sample_0, torch.zeros_like(mean_sample_0),
                               log_var_sample_0,
                               torch.zeros_like(log_var_sample_0))
        kl_loss_1 = compute_kl(mean_sample_1, torch.zeros_like(mean_sample_1),
                               log_var_sample_1,
                               torch.zeros_like(log_var_sample_1))
        # sum over latent dimensions, mean over batch dimension
        kl_loss = torch.cat([kl_loss_0 + kl_loss_1], dim=0).sum(dim=1).mean()
        kl_per_dim = torch.cat([kl_loss_0 + kl_loss_1], dim=0).mean(dim=0)
        vae_loss = self.beta * kl_loss + recon_loss
        infos = {'kl_normal_adagvae': kl_loss.item(),
                 'total_loss': vae_loss.item(),
                 'kl_per_dim': kl_per_dim,
                 'mse_recon_loss': mse_recon_loss.item(),
                 'bce_recon_loss': bce_recon_loss.item()}
        return vae_loss, infos, x_recon

    def loss_f(self, x, x_recon, mu, logvar, reduce=True, return_all=False):
        raise NotImplementedError


def compute_kl(mu0: torch.tensor, mu1: torch.tensor, logvar_0: torch.tensor,
               logvar_1: torch.tensor):
    var_1 = logvar_0.exp()
    var_2 = logvar_1.exp()
    return var_1 / var_2 + (mu1 - mu0) ** 2 / var_2 - 1 + logvar_1 - logvar_0

def aggregate_s_known(mu, logvar, new_mu, new_logvar, s_mask):
    return torch.where(s_mask, new_mu, mu), torch.where(s_mask, new_logvar, logvar) 

def aggregate_argmax(mu, logvar, new_mu, new_log_var, kl_per_point):
    """Argmax aggregation with adaptive k.
    The bottom k dimensions in terms of distance are not averaged. K is
    estimated adaptively by binning the distance into two bins of equal width.
    Args:
      mu: Mean of the encoder distribution for the original image.
      logvar: Logvar of the encoder distribution for the original image.
      new_mu: Average mean of the encoder distribution of the pair of images.
      new_log_var: Average logvar of the encoder distribution of the pair of
        images.
      kl_per_point: Distance between the two encoder distributions.
    Returns:
      Mean and logvariance for the new observation.
    """

    # mimic tf histogram_fixed_width_bins with n_bins=2
    kl_middle = (kl_per_point.max(dim=1, keepdim=True).values +
                 kl_per_point.min(dim=1, keepdim=True).values) / 2
    mask = torch.zeros_like(kl_per_point, dtype=torch.bool)
    mask[kl_per_point > kl_middle] = True

    mu_averaged = torch.where(mask, mu, new_mu)
    logvar_averaged = torch.where(mask, logvar, new_log_var)

    return mu_averaged, logvar_averaged


class AdaGVAE_K_Known(AdaGVAE):
    def aggregate_argmax(mu, logvar, new_mu, new_log_var, kl_per_point):
        """
        Assume k is known and fixed (k=1)
        """
        kl_bottom_idxs = torch.argmin(kl_per_point, dim=-1) # assume k is known 
        mask = torch.zeros_like(kl_per_point, dtype=torch.bool)
        mask[torch.arange(kl_per_point.shape[0]), kl_bottom_idxs] = True

        mu_averaged = torch.where(mask, new_mu, mu)
        logvar_averaged = torch.where(mask, new_log_var, logvar)

        return mu_averaged, logvar_averaged

    
# implementation inspired by tf code from Morioka
class PCL(Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ai = torch.nn.Parameter(torch.ones(1, self.number_latents))
        self.bi = torch.nn.Parameter(-torch.ones(1, self.number_latents))
        self.ci = torch.nn.Parameter(torch.zeros(1, self.number_latents))
        self.m = torch.nn.Parameter(torch.zeros(1))

    def regression_function(self, h1, h2):
        # Build r(y) ----------------------------------------------
        # sum_i |ai*hi(y1) + bi*hi(y2) + ci| - (di*hi(y1) + ki)^2 + mi
        #         ----------------------------   ----------------
        #                       Q                       Qbar
        # ai and ki are fixed to 0 because they have indeterminacy with scale
        # and bias of hi.
        # [ai, bi, ci] are initialized by [1,-1,0].
        Q = (self.ai * h1 + self.bi * h2 + self.ci).abs().sum(dim=1)
        Q_bar = (h1 ** 2).sum(dim=1)  # second part in 4.1 in pcl paper
        logits = -Q + Q_bar + self.m
        return logits

    def loss_f(self, latents):
        assert latents.shape[0] % 2 == 0
        x_t_minus1 = latents[::2]
        x_t = latents[1::2]
        batch_size = x_t_minus1.shape[0]
        x_t_minus1_permuted = x_t_minus1[range(-1, batch_size - 1)]
        logits_true = self.regression_function(x_t, x_t_minus1)
        logits_false = self.regression_function(x_t, x_t_minus1_permuted)
        logits = torch.cat([logits_true, logits_false])
        labels_cl = torch.cat(
            [torch.ones(batch_size),  # true, positive pair
             torch.zeros(batch_size)  # false, negative pair
             ]).to(x_t.device)
        pcl_loss = F.binary_cross_entropy_with_logits(logits, labels_cl)

        # bookkeeping
        n_correct = ((logits > 0).float() == labels_cl).float().sum().item()
        n_total = logits.shape[0]
        infos = {'pcl_accuracy': n_correct / n_total,
                 'pcl_loss': pcl_loss.item()}
        return pcl_loss, infos
    
    def repn_fn(self, x, key=None): 
        return self(x)

    def forward(self, x):
        latents = super().forward(x)[:, :self.number_latents]
        return latents

def reconstruction_loss(x, x_recon, distribution, batch_size, reduction='sum'):
    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, reduction=reduction).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = torch.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, reduction=reduction).div(batch_size)
    else:
        recon_loss = None
    if reduction == 'none':
        recon_loss = recon_loss.flatten(-3).sum(-1)
    return recon_loss

def compute_ent_normal(logvar):
    return 0.5 * (logvar + np.log(2 * np.pi * np.e))

def compute_cross_ent_normal(mu, logvar):
    return 0.5 * (mu ** 2 + torch.exp(logvar)) + np.log(np.sqrt(2 * np.pi))


def get_model(model_name, number_classes, number_channels, number_latents,
              args, dataset=None):
    if model_name == 'pcl':
        model = PCL(number_latents=number_latents,
                    number_channels=number_channels)
    elif model_name == 'betavae':
        model = BetaVAE(number_latents=number_latents,
                        number_channels=number_channels,
                        beta=args.vae_beta)
    elif model_name == 'slowvae':
        model = SlowVAE(number_latents=number_latents,
                        number_channels=number_channels,
                        gamma=args.slowvae_gamma, beta=args.vae_beta,
                        rate_prior=args.slowvae_rate)
    elif model_name == 'adagvae':
        model = AdaGVAE(number_latents=number_latents,
                        number_channels=number_channels,
                        beta=args.vae_beta)
    elif model_name == 'adagvae_k_known':
        model = AdaGVAE_K_Known(number_latents=number_latents,
                        number_channels=number_channels,
                        beta=args.vae_beta)
    elif model_name == 'gvae': 
        model = GVAEModel(number_latents=number_latents, 
                          number_channels=number_channels, 
                          beta=args.vae_beta)
    elif model_name == 'mlvae': 
        model = MLVAEModel(number_latents=number_latents, 
                           number_channels=number_channels, 
                           beta=args.vae_beta)
    else:
        raise Exception(f'Model {args.model} is not defined')
    return model