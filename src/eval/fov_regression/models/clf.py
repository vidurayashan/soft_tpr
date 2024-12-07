import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
import abc
from src.shared.constants.loss_fn import BCE, SOFT_HINGE
from src.shared.components import MLP, ModularMLP

class BaseClf(nn.Module, abc.ABC): 
    def __init__(self, net, n_fillers_per_role: np.array, loss_type: str=BCE, 
                 filler_weights: torch.Tensor=None, role_weights: torch.Tensor=None) -> None: 
        super().__init__()
        self.net = net 
        self.n_fillers_per_role = n_fillers_per_role
        self.n_factors = len(n_fillers_per_role)
        self.loss_type = loss_type
        self.filler_weights = self.init_filler_weights(filler_weights).cuda()  # (1, NF)
        self.role_weights = self.init_role_weights(role_weights).cuda() # (1, NR) 
        self.weights = self.get_weights() 
        self.eps = 1e-4

    def get_weights(self) -> torch.Tensor: 
        expanded_role_weights = torch.repeat_interleave(self.role_weights, 
                                                          repeats=torch.from_numpy(self.n_fillers_per_role).cuda(),
                                                          dim=1).cuda() # (1, NF)
        return self.filler_weights * expanded_role_weights

    def init_role_weights(self, role_weights: torch.Tensor) -> torch.Tensor: 
        if role_weights is None: 
            return torch.ones(size=(1, len(self.n_fillers_per_role)), requires_grad=False) 
        assert len(role_weights.shape) == 2 and role_weights.shape[0] == 1, f'Role weights incorrect shape {role_weights.shape}'
        return role_weights

    def init_filler_weights(self, filler_weights: torch.Tensor) -> torch.Tensor: 
        if filler_weights is None: 
                return torch.ones(size=(1, sum(self.n_fillers_per_role)), requires_grad=False)
        assert len(filler_weights.shape) == 2 and filler_weights.shape[0] == 1, f'Filler weights incorrect shape {filler_weights.shape}'
        return filler_weights
    
    def compute_loss(self, logits: torch.Tensor, tgts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: 
        if self.loss_type == BCE: 
            loss = F.binary_cross_entropy_with_logits(input=logits, target=tgts, reduction='none')
            #print(f'Self weights is {self.weights}')
            weighted_loss = (loss * 1/(self.weights.cuda() + self.eps)).mean() 
            return loss.mean(), weighted_loss
        else: 
            assert self.loss_type == SOFT_HINGE, f'Unrecognised loss type'
            split_logits = logits.split(split_size=list(self.n_fillers_per_role), dim=1)
            split_tgts = tgts.split(split_size=list(self.n_fillers_per_role), dim=1) 
            losses = []
            for per_fac_logits, per_fac_tgts in zip(split_logits, split_tgts): 
                loss = F.multilabel_soft_margin_loss(input=per_fac_logits, target=per_fac_tgts, reduction='none') # (N, N_fillers_f)
                losses.append(loss)
            losses = torch.concatenate(losses, dim=1) # (N, N_fillers)
            return losses.mean(), torch.mean(1/(self.weights.cuda() + self.eps) * losses)
            
    def make_state(self, latent_reps: torch.Tensor, preds: torch.Tensor, logits: torch.Tensor, 
                   tgt: torch.Tensor) -> Dict: 
        return {'latent_reps': latent_reps, 'preds': preds, 'logits': logits, 'tgt': tgt}
    
    def get_preds(self, logits: torch.Tensor, per_factor_pred: bool=True) -> torch.Tensor: 
        if per_factor_pred: 
            split_logits = torch.split(logits, split_size_or_sections=list(self.n_fillers_per_role), dim=1) 
            preds = []
            for n_fillers_per_fac, per_factor_logits in zip(self.n_fillers_per_role, split_logits): 
                preds.append(
                    F.one_hot(torch.argmax(per_factor_logits, dim=1), num_classes=n_fillers_per_fac) # (N, 1)
                )
            preds = torch.concatenate(preds, dim=1) 
            return preds
        return (F.sigmoid(logits) > 0.5)
    

    def get_acc_logs(self, preds: torch.Tensor, tgts: torch.Tensor) -> Dict: 
        mask = ((preds == tgts.to(dtype=torch.bool)))

        n_factors_correct = torch.sum((preds & (preds & mask)).flatten())

        n_combs_correct = torch.sum(torch.sum(mask, dim=1) == tgts.shape[1])
        acc_factors = n_factors_correct / torch.sum(tgts.flatten())
        acc_combs = n_combs_correct / tgts.shape[0]

        acc_logs = {'acc_factors': acc_factors, 'acc_combs': acc_combs}
        return acc_logs     

    def forward(self, latent_reps: torch.Tensor, tgts: torch.Tensor) -> torch.Tensor: 
        logits = self.net(latent_reps)
        tgts = tgts.to(dtype=torch.float32)

        loss, weighted_loss = self.compute_loss(logits=logits, tgts=tgts)
        loss_logs = {'total_loss': weighted_loss, 'unweighted_loss': loss.mean()}

        preds = self.get_preds(logits)
        acc_logs = self.get_acc_logs(preds=preds, tgts=tgts)

        return {'loss': {**loss_logs, **acc_logs}, 'state': self.make_state(latent_reps=latent_reps,
                                                                             preds=preds, 
                                                                             logits=logits,
                                                                             tgt=tgts)}
    
    @staticmethod 
    def get_filler_weights(one_hot_tgts: torch.Tensor, n_fillers_per_role: np.array=None) -> torch.Tensor: 
        if n_fillers_per_role is None: 
            # return proportion of each filler relative to number of all fillers
            return (torch.sum(one_hot_tgts, dim=0) / torch.sum(one_hot_tgts)).unsqueeze(0)
        # otherwise, return proportion of each filler relative to the roles 
        props = list(map(lambda x: torch.sum(x, dim=0)/torch.sum(x), 
                         torch.split(one_hot_tgts, split_size_or_sections=list(n_fillers_per_role), dim=1)))
        return torch.concatenate(props, dim=0).unsqueeze(0)

    @staticmethod 
    def serialise_one_hot(one_hot: torch.Tensor, one_hot_pred_to_factor: Dict, 
                          one_hot_pred_to_serialised: Dict, pred: bool) -> List[List[str]]: 
        prefix = 'Pred' if pred else 'Actual'

        idxs_per_obs = list(
            map(lambda preds_per_obs: torch.argwhere(preds_per_obs == True),
                torch.unbind(one_hot, dim=0)),
        )
        
        serialised_one_hot = list(
            map(lambda pred_idxs_one_obs, one_hot_pred_to_factor=one_hot_pred_to_factor, 
                one_hot_pred_to_serialised=one_hot_pred_to_serialised, prefix=prefix: 
                    [f'Factor: {one_hot_pred_to_factor[pred_idxs_one_obs[i].item()]}\n' + 
                        f'{prefix}: {one_hot_pred_to_serialised[pred_idxs_one_obs[i].item()]}'
                        for i in range(pred_idxs_one_obs.shape[0])
                    ], 
            idxs_per_obs)
        )
        return serialised_one_hot

    @staticmethod 
    def serialise_preds(preds: torch.Tensor, one_hot_pred_to_factor: Dict, 
                        one_hot_pred_to_serialised: Dict) -> List[List[str]]: 
        return BaseClf.serialise_one_hot(one_hot=preds, one_hot_pred_to_factor=one_hot_pred_to_factor,
                                         one_hot_pred_to_serialised=one_hot_pred_to_serialised, 
                                         pred=True) 
    
    @staticmethod 
    def serialise_tgts(tgts: torch.Tensor, one_hot_pred_to_factor: Dict, 
                       one_hot_pred_to_serialised: Dict) -> List[List[str]]: 
        return BaseClf.serialise_one_hot(one_hot=tgts, one_hot_pred_to_factor=one_hot_pred_to_factor,
                                         one_hot_pred_to_serialised=one_hot_pred_to_serialised, 
                                         pred=False) 
    
    @staticmethod 
    def display_predictions(preds: torch.Tensor, tgts: torch.Tensor, one_hot_pred_to_factor: Dict, 
                            one_hot_pred_to_serialised: Dict, n_to_display: int=20) -> None: 
        n_to_display = min(n_to_display, preds.shape[0])
        preds = preds[:n_to_display]
        tgts = tgts[:n_to_display]

        print(f'Preds has shape {preds.shape}, tgts {tgts.shape}')

        serialised_preds = BaseClf.serialise_preds(preds=preds, one_hot_pred_to_factor=one_hot_pred_to_factor, 
                                                   one_hot_pred_to_serialised=one_hot_pred_to_serialised)
        serialised_tgts = BaseClf.serialise_tgts(tgts=tgts, one_hot_pred_to_factor=one_hot_pred_to_factor,
                                                 one_hot_pred_to_serialised=one_hot_pred_to_serialised)
        prefix_appender = lambda pt: '' if pt[0].split(':')[-1] == pt[1].split(':')[-1] else '****INCORRECT****\t'
        for pred, tgt in zip(serialised_preds, serialised_tgts): 
            to_print = '\n'.join([prefix_appender((p,t)) + p + '\t' + t.split('\n')[1] for p, t in zip(pred, tgt)])
            print(to_print + '\n')

    def get_output_from_incorrect_preds(self, clf_dataloader: DataLoader, model,
                                        one_hot_pred_to_serialised: Dict, factor_idx_to_serialised: Dict,
                                        use_encoded_input_as_clf_input: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        incorrect_inputs = []
        incorrect_latents = []
        incorrect_preds = []
        incorrect_logits = []
        incorrect_tgts = []
        incorrect_factor_freq = {fac_name: 0 for fac_name in factor_idx_to_serialised.values()} # TODO: change so that dataloader contains dataset attr, and then we get one_hot_pred and factor_idx from metadatamap

        with torch.no_grad(): 
            self.eval()
            for x, tgts in clf_dataloader: 
                tgts = tgts.cuda() 
                x = x.cuda() 
                if use_encoded_input_as_clf_input:
                    latents = model.encoder(x)['rep']
                    clf_out = self(latents, tgts)
                else: 
                    clf_out = self(x, tgts)
                #print(f'Clf out is {clf_out}')
                #acc_factors = clf_out['loss']['acc_factors']
                #acc_combs = clf_out['loss']['acc_combs']
                #print(f'Acc factors acc combs is {acc_factors}, {acc_combs}')
                preds, logits = clf_out['state']['preds'], clf_out['state']['logits']
                incorrect_idxs = torch.argwhere(
                    ((preds != tgts.to(dtype=torch.bool)))
                )
                if incorrect_idxs.numel() != 0:
                    incorrect_obs_idxs, unique_idxs = torch.unique(incorrect_idxs[:, 0], dim=0, return_inverse=True)
                    incorrect_fac_idxs = incorrect_idxs[unique_idxs, 1]

                    incorrect_inputs.extend(x[incorrect_obs_idxs].unbind(0))
                    if use_encoded_input_as_clf_input:
                        incorrect_latents.extend(latents[incorrect_obs_idxs].unbind(0))
                    incorrect_preds.extend(preds[incorrect_obs_idxs].unbind(0))
                    incorrect_logits.extend(logits[incorrect_obs_idxs].unbind(0))
                    incorrect_tgts.extend(tgts[incorrect_obs_idxs].unbind(0))

                    incorrect_facs, counts = torch.unique(incorrect_fac_idxs, return_counts=True) 
                    for i in range(incorrect_facs.shape[0]): 
                        incorrect_factor_freq[one_hot_pred_to_serialised[incorrect_facs[i].item()]] += counts[i].item()

            if incorrect_inputs != []:
                incorrect_inputs = torch.stack(incorrect_inputs, dim=0) # (N, 5)
                if use_encoded_input_as_clf_input: 
                    incorrect_latents = torch.stack(incorrect_latents, dim=0)
                incorrect_preds = torch.stack(incorrect_preds, dim=0)
                incorrect_logits = torch.stack(incorrect_logits, dim=0)
                incorrect_tgts = torch.stack(incorrect_tgts, dim=0)

            return incorrect_inputs, incorrect_latents, incorrect_preds, incorrect_logits, incorrect_tgts, incorrect_factor_freq


class Clf(BaseClf): 
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: List[int], n_fillers_per_role: np.array=None,
                 loss_type: str='bce', filler_weights: torch.Tensor=None, role_weights: torch.Tensor=None,
                force_per_factor_pred: bool=False) -> None: 
        super().__init__(net=MLP(in_dim=in_dim, out_dim=out_dim, hidden_dims=hidden_dims), 
              n_fillers_per_role=n_fillers_per_role, loss_type=loss_type,
              filler_weights=filler_weights, role_weights=role_weights) 
        self.force_per_factor_pred = force_per_factor_pred
        self.kwargs_for_loading = {
            'in_dim': in_dim, 
            'out_dim': out_dim, 
            'hidden_dims': hidden_dims, 
            'n_fillers_per_role': n_fillers_per_role,
            'filler_weights': filler_weights,
            'role_weights': role_weights,
            'force_per_factor_pred': force_per_factor_pred, 
            'loss_type': loss_type,
        }
        
    def get_preds(self, logits: torch.Tensor) -> torch.Tensor: 
        return super().get_preds(logits=logits, per_factor_pred=self.force_per_factor_pred)
    
class ModularClf(BaseClf): 
    def __init__(self, in_dim: int, hidden_dims: List[int], 
                 n_fillers_per_role: np.array=None, loss_type: str='bce',
                 filler_weights: torch.Tensor=None, role_weights: torch.Tensor=None) -> None: 
        super().__init__(net=ModularMLP(in_dim=in_dim, out_dims=list(n_fillers_per_role), hidden_dims=hidden_dims),
                         n_fillers_per_role=n_fillers_per_role, loss_type=loss_type, 
                         filler_weights=filler_weights, role_weights=role_weights)
        self.kwargs_for_loading = {
            'in_dim': in_dim, 
            'hidden_dims': hidden_dims, 
            'n_fillers_per_role': n_fillers_per_role,
            'filler_weights': filler_weights,
            'role_weights': role_weights, 
            'loss_type': loss_type,
        }