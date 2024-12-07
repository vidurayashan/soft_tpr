import torch.nn as nn

""" Downstream model architecture for abstract visual reasoning uses the same specification as:

@inproceedings{
schott2022visual,
title={Visual Representation Learning Does Not Generalize Strongly Within the Same Domain},
author={Lukas Schott and Julius Von K{\"u}gelgen and Frederik Tr{\"a}uble and Peter Vincent Gehler and Chris Russell and Matthias Bethge and Bernhard Sch{\"o}lkopf and Francesco Locatello and Wieland Brendel},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=9RUHPlladgh}
}
"""
class ReadOutMLP(nn.Sequential):
    def __init__(self, in_features: int, out_features: int,
                 d1: int=256, d2: int=256, d3: int=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, d1),
            nn.ReLU(),
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Linear(d2, d2),
            nn.ReLU(),
            nn.Linear(d2, d2),
            nn.ReLU(),
            nn.Linear(d2, d3),
             nn.ReLU(),
            nn.Linear(d3, out_features),
        )
        self.kwargs_for_loading = {
            'in_features': in_features, 
            'out_features': out_features, 
            'd1': d1,
            'd2': d2, 
            'd3': d3,
        }