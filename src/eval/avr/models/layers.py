""" 
Code below adapted from the open-source implementation of the Wild Relation Network at: https://github.com/mikomel/wild-relation-network/blob/main/wild_relation_network/layers.py

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
from typing import List
import torch
from torch import nn

from src.eval.avr.shared import *

class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
class LinearBNReLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(LinearBNReLU, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        shape = x.shape
        x = x.flatten(0, -2)
        x = self.bn(x)
        x = x.view(shape)
        x = self.relu(x)
        return x
class DeepLinearBNReLU(nn.Module):
    def __init__(self, depth: int, in_dim: int, hidden_dim: int, out_dim: int):
        super(DeepLinearBNReLU, self).__init__()
        self.layers = nn.Sequential(*[
            LinearBNReLU(d1, d2)
            for d1, d2 in zip([in_dim] + [hidden_dim]*depth, [hidden_dim]*depth + [out_dim])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
class GroupObjectsIntoPairs(nn.Module):
    def __init__(self, num_objects: int):
        super().__init__()
        self.num_objects = num_objects

    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects, object_size = objects.size()
        return torch.cat([
            objects.unsqueeze(1).repeat(1, num_objects, 1, 1),
            objects.unsqueeze(2).repeat(1, 1, num_objects, 1)
        ], dim=3).view(batch_size, num_objects ** 2, 2 * object_size)

    def num_pairs(self):
        return self.num_objects ** 2
class GroupObjectsIntoPairsWith(nn.Module):
    def forward(self, objects: torch.Tensor, object: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            objects,
            object.unsqueeze(1).repeat(1, 8, 1)
        ], dim=2)
class GroupObjectsIntoTriples(nn.Module):
    def __init__(self, num_objects: int):
        super().__init__()
        self.num_objects = num_objects

    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects, object_size = objects.size()
        return torch.cat([
            objects.unsqueeze(1).repeat(1, num_objects, 1, 1).unsqueeze(2).repeat(1, 1, num_objects, 1, 1),
            objects.unsqueeze(1).repeat(1, num_objects, 1, 1).unsqueeze(3).repeat(1, 1, 1, num_objects, 1),
            objects.unsqueeze(2).repeat(1, 1, num_objects, 1).unsqueeze(3).repeat(1, 1, 1, num_objects, 1)
        ], dim=4).view(batch_size, num_objects ** 3, 3 * object_size)

    def num_triples(self) -> int:
        return self.num_objects ** 3
class GroupObjectsIntoTriplesWith(nn.Module):
    def forward(self, objects: torch.Tensor, object: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects, object_size = objects.size()
        return torch.cat([
            objects.unsqueeze(1).repeat(1, num_objects, 1, 1),
            objects.unsqueeze(2).repeat(1, 1, num_objects, 1),
            object.unsqueeze(1).unsqueeze(2).repeat(1, num_objects, num_objects, 1)
        ], dim=3).view(batch_size, num_objects ** 2, 3 * object_size)
class TagPanelEmbeddings(nn.Module):
    """ Tags panel embeddings of 3x3 Raven's Progressive Matrices (RPMs) with their absolute coordinates. """

    def forward(self, panel_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Concatenates a one-hot encoded vector to each RPM panel.
        The concatenated vector indicates panel absolute position in the RPM.
        :param panel_embeddings: a tensor of shape (batch_size, 16, embedding_size)
        :return: a tensor of shape (batch_size, 16, embedding_size + 9)
        """
        batch_size = panel_embeddings.shape[0]
        tags = torch.zeros((N_ANS + N_CONTEXT, N_CONTEXT + 1), device=panel_embeddings.device).type_as(panel_embeddings)
        tags[:N_CONTEXT, :8] = torch.eye(8, device=panel_embeddings.device).type_as(panel_embeddings)
        tags[N_CONTEXT:, 8] = torch.ones(N_ANS, device=panel_embeddings.device).type_as(panel_embeddings)
        tags = tags.expand((batch_size, -1, -1))
        return torch.cat([panel_embeddings, tags], dim=2)