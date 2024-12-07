import torch 
import math 
import torch.nn as nn

from typing import List

class View(nn.Module): 
    def __init__(self, *size): 
        super(View, self).__init__() 
        self.size = size 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return x.view(*self.size)

class MLP(nn.Module): 
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: List[int]=None) -> None: 
        super(MLP, self).__init__()
        self.net = []
        self.out_dim = out_dim 
        self.in_dim = in_dim
        if hidden_dims is not None: 
            hidden_dims = [in_dim] + hidden_dims + [out_dim]
            for (h0, h1) in zip(hidden_dims, hidden_dims[1:]):
                self.net.extend([nn.Linear(h0, h1),
                                  nn.ReLU()])
            self.net.pop()
        else: 
            self.net = [nn.Linear(in_features=in_dim, out_features=out_dim)]    
        self.net = nn.Sequential(*self.net)
        self.kwargs_for_loading = {'in_dim': in_dim, 
                                   'out_dim': out_dim, 
                                   'hidden_dims': hidden_dims}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class ModularMLP(nn.Module): 
    def __init__(self, in_dim: int, out_dims: List[int], hidden_dims: List[int], modular_input: bool=False) -> None: 
        super().__init__()
        self.n_modules = len(out_dims)
        self.nets = nn.ParameterList([MLP(in_dim=in_dim, out_dim=out_dims[i], hidden_dims=hidden_dims) 
                                         for i in range(self.n_modules)])
        self.modular_input = modular_input
    
    def forward(self, x) -> torch.Tensor: 
        if self.modular_input: 
            assert isinstance(x, list) or isinstance(x, tuple), f'Modular input with type list expected, but input is of type {type(x)}'
            stacked_output = [net(x_module) for (net, x_module) in zip(self.nets, x)]
        else: 
            stacked_output = [net(x) for net in self.nets]
        return torch.concatenate(stacked_output, dim=1) 
    
class EmbedLayer(nn.Module): 
    def __init__(self, output_dim: int,
                 latent_dim: int) -> None: 
        super().__init__() 
        self.n_embeddings = latent_dim
        self.per_embed_dim = math.ceil(output_dim/self.n_embeddings)
        self.output_dim = self.per_embed_dim*self.n_embeddings 
        self.embeddings = nn.Embedding(num_embeddings=self.n_embeddings, 
                                       embedding_dim=self.per_embed_dim).requires_grad_(False)
        nn.init.orthogonal_(self.embeddings.weight) # ensures that embeddings easily distinguishable
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor: 
        # (N, n_embeddings, 1)
        # (1, n_embeddings, embed_dim)
        multiplied_embeddings = latent.unsqueeze(-1) * self.embeddings.weight.unsqueeze(0)
        concatenated = multiplied_embeddings.view(-1, self.output_dim)
        return concatenated 