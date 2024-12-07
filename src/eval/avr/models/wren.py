""" 
Code below adapted from the open-source implementation of the Wild Relation Network at: https://github.com/mikomel/wild-relation-network/blob/main/wild_relation_network/wild_relation_network.py

@inproceedings{santoro2017simple,
  title={A simple neural network module for relational reasoning},
  author={Santoro, Adam and Raposo, David and Barrett, David G and Malinowski, Mateusz and Pascanu, Razvan and Battaglia, Peter and Lillicrap, Timothy},
  booktitle={Advances in neural information processing systems},
  pages={4967--4976},
  year={2017}
}

@inproceedings{santoro2018measuring,
  title={Measuring abstract reasoning in neural networks},
  author={Santoro, Adam and Hill, Felix and Barrett, David and Morcos, Ari and Lillicrap, Timothy},
  booktitle={International Conference on Machine Learning},
  pages={4477--4486},
  year={2018}
}
""" 

import torch 
import torch.nn as nn
import torch.nn.functional as F
from src.shared.constants import *
from src.eval.avr.shared import *
from src.eval.avr.models.layers import GroupObjectsIntoPairs, GroupObjectsIntoPairsWith, TagPanelEmbeddings, Identity, \
    GroupObjectsIntoTriples, GroupObjectsIntoTriplesWith
from src.eval.avr.models import relation_network

class WReN(nn.Module):
    """
    Wild Relation Network (WReN) [1] for solving Raven's Progressive Matrices.
    The originally proposed model uses a Relation Network (RN) [2] which works on object pairs.
    This implementation allows to extend the RN to work on object triples (by setting use_object_triples=True).
    Using larger tuples is impractical, as the memory requirement grows exponentially,
    with complexity O(num_objects ^ rn_tuple_size).
    After extension to triples, the model resembles the Logic Embedding Network (LEN) [3].

    [1] Santoro, Adam, et al. "Measuring abstract reasoning in neural networks." ICML 2018
    [2] Santoro, Adam, et al. "A simple neural network module for relational reasoning." NeurIPS 2017
    [3] Zheng, Kecheng, Zheng-Jun Zha, and Wei Wei. "Abstract reasoning with distracting features." NeurIPS 2019
    """

    def __init__(self, embedding_fn, 
                 embedding_dim: int, 
                 hidden_size_g: int = None, 
                 hidden_size_f: int = None,
                 use_object_triples: bool = False, use_layer_norm: bool = False):
        """
        Initializes the WReN model.
        :param num_channels: number of convolutional kernels in each CNN layer
        :param use_object_triples: flag indicating whether to group objects into triples (by default object pairs are considered)
        in the Relation Network submodule. Use False to reproduce WReN and True to reproduce LEN.
        :param use_layer_norm: flag indicating whether layer normalization should be applied after
        the G submodule of RN.
        """
        super(WReN, self).__init__()
        if use_object_triples:
            self.group_objects = GroupObjectsIntoTriples(num_objects=8)
            self.group_objects_with = GroupObjectsIntoTriplesWith()
        else:
            self.group_objects = GroupObjectsIntoPairs(num_objects=8)
            self.group_objects_with = GroupObjectsIntoPairsWith()
        
        self.embedding_fn = embedding_fn
        self.object_size = embedding_dim

        self.object_tuple_size = (3 if use_object_triples else 2) * (self.object_size + 9)
        self.tag_panel_embeddings = TagPanelEmbeddings()
        
        if hidden_size_g is None: 
            hidden_size_g = self.object_tuple_size 
        if hidden_size_f is None: 
            hidden_size_f = self.object_tuple_size
        
        self.g = relation_network.G(
            depth=3,
            in_size=self.object_tuple_size,
            hidden_size=hidden_size_g,
            out_size=self.object_tuple_size,
            use_layer_norm=False
        )
        self.norm = nn.LayerNorm(self.object_tuple_size) if use_layer_norm else Identity()
        self.f = relation_network.F(
            depth=2,
            object_size=self.object_tuple_size,
            hidden_size=hidden_size_f,
            out_size=1
        )

    def forward(self, x: torch.Tensor, labels :torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the WReN model.
        :param x: a tensor with shape (batch_size, num_panels, height, width). num_panels is assumed
        to be 16, as the model was designed to solve RPMs from the PGM dataset.
        :return: a tensor with shape (batch_size, num_answers). num_answers is always equal to 8,
        which is the number of answers for each RPM in PGM.
        """
        # assume features (B, (n_ans + n_context), 3, H, W)
        batch_size = labels.shape[0]
        x = x.view((-1, *x.shape[-3:]))
        with torch.no_grad():
            x = self.embedding_fn(x)
        x = x.view(batch_size, N_CONTEXT+N_ANS, self.object_size)
        x = self.tag_panel_embeddings(x)
        context_objects = x[:, :N_CONTEXT, :] # assume first N_CONTEXT objects context, last the choices
        choice_objects = x[:, N_CONTEXT:, :]
        context_pairs = self.group_objects(context_objects)
        context_g_out = self.g(context_pairs)
        f_out = torch.zeros(batch_size, N_ANS, device=x.device).type_as(x)
        for i in range(N_ANS):
            context_choice_pairs = self.group_objects_with(context_objects, choice_objects[:, i, :])
            context_choice_g_out = self.g(context_choice_pairs)
            relations = context_g_out + context_choice_g_out
            relations = self.norm(relations)
            f_out[:, i] = self.f(relations).squeeze()
        
        loss = F.binary_cross_entropy_with_logits(
            input=f_out, target=labels.float()
        )
        preds = torch.argmax(f_out, 1)
        acc = (preds == labels.argmax(1)).sum()/preds.shape[0]
        return {'loss': loss, 'acc': acc}


import math 
def lecun_normal_(tensor: torch.Tensor) -> torch.Tensor: 
    if not isinstance(tensor, nn.Linear):
        return 
    input_size = tensor.weight.shape[-1]
    std = math.sqrt(1/input_size)
    with torch.no_grad(): 
        return tensor.weight.normal_(-std, std)
