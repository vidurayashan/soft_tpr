"""
The below code is adapted from the code for the paper 'Commutative Lie Group VAE for Disentanglement Learning' (ICML, 2021)
at: https://github.com/zhuxinqimac/CommutativeLieGroupVAE-Pytorch/blob/main/metrics/aggregator.py

@inproceedings{Xinqi_liegroupvae_icml21,
author={Xinqi Zhu and Chang Xu and Dacheng Tao},
title={Commutative Lie Group VAE for Disentanglement Learning},
booktitle={ICML},
year={2021}
}
"""
import torch
import numpy as np
from src.data.datasets import DisLibDataset
from src.eval.dis.metrics import *
class MetricAggregator:
    def __init__(self, val_dataset: DisLibDataset, n_points: int, 
                 random_state1: np.random.RandomState, 
                 random_state2: np.random.RandomState, 
                 use_multidim_latents: bool=False, discretise_soft_tpr_repn: bool=True, 
                 verbose: bool=True,
                 use_beta: bool=True, use_mig: bool=True, use_dci: bool=True, 
                 use_factor: bool=True):
        """ Helper class to compute disentanglement metrics

        Args:
            val_dataset (Dataset): torch Dataset object on which to evaluate metrics
            n_points (int): Number of points to use in metric calculation
            model (nn.Module): PyTorch model to evaluate
            verbose (bool): If True print verbosely
        """
        if isinstance(val_dataset, torch.utils.data.Subset): 
            val_dataset = val_dataset.dataset 
        self.val_dataset = val_dataset
        self.n_points = n_points
        self.verbose = verbose
        self.random_state1 = random_state1
        self.random_state2 = random_state2 
        self.use_multidim_latents = use_multidim_latents
        self.discretise_soft_tpr_repn = discretise_soft_tpr_repn
        self.metrics = self._init_metrics(use_beta, use_mig, use_dci, use_factor)

    def _init_metrics(self, use_beta: bool=True, use_mig:bool=True, use_dci:bool=True, use_factor: bool=True):
        metrics = []
        
        if use_beta: 
            hig = BetaVAEMetric(ds=self.val_dataset, 
                                num_points=self.n_points, 
                                random_state1=self.random_state1,
                                random_state2=self.random_state2, 
                                use_multidim_latents=self.use_multidim_latents)
            metrics.append(hig)
        if use_mig: 
            mig = MigMetric(ds=self.val_dataset, 
                            num_points=self.n_points, 
                            random_state=self.random_state1,
                            use_multidim_latents=self.use_multidim_latents)
            metrics.append(mig)
        if use_dci: 
            dci = DciMetric(dataset=self.val_dataset, 
                            n_points=min(self.n_points, 10000), 
                            random_state1=self.random_state1,
                            random_state2 = self.random_state2, 
                            use_multidim_latents=self.use_multidim_latents,
                            use_discrete_repn=self.discretise_soft_tpr_repn)
            metrics.append(dci) 
        if use_factor: 
            fac = FactorVAEMetric(self.val_dataset, 
                                n_train=self.n_points, 
                                random_state1=self.random_state1,
                                random_state2=self.random_state2,
                                use_discrete_repn=self.discretise_soft_tpr_repn, n_eval=self.n_points//2, batch_size=64, n_var_est=self.n_points, 
                                use_multidim_latents=self.use_multidim_latents)
            metrics.append(fac)
        
        #mod = Modularity(self.val_dataset, num_points=10000)
        #sap = SapMetric(self.val_dataset, num_points=10000)

        return metrics

    def __call__(self, repn_fn, eval: bool=False):
        import gc
        with torch.no_grad():
            outputs = {}
            for metric in self.metrics:
                if self.verbose:
                    print("Computing metric: {}".format(metric))
                outputs.update(metric(repn_fn, eval))
                gc.collect()
            if self.verbose: 
                print('outputs:', outputs)
            return outputs