import torch
import torch.nn as nn
import torch.nn.functional as F

def get_n_layers(curr_w: int, desired_w: int, p: int, k_size: int, s: int, l: int) -> int: 
    if desired_w >= curr_w: 
        if desired_w > curr_w: 
            return -1
        return l 
    new_w = ((curr_w-k_size+2*p)/s) + 1 
    return get_n_layers(curr_w=new_w, desired_w=desired_w, p=p, k_size=k_size, s=s, l=l+1)

def init_embeddings(init_orth: bool, weights: torch.Tensor) -> None: 
    n_embed = weights.shape[0]
    if init_orth: 
        torch.nn.init.orthogonal_(weights)
    else: 
        torch.nn.init.normal_(weights, 0, 1/n_embed)

def init_embeddings_truncated_identity(weights: torch.Tensor) -> torch.Tensor:
    torch.nn.init.eye_(weights)
    
""" 
The below code for initialisation was taken directly from the following Github repo:
https://github.com/YannDubs/disentangling-vae/blob/master/disvae/utils/initialization.py
""" 

def gaussian_kls(mu, logvar, mean=False):

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())

    if mean:
        reduce = lambda x: torch.mean(x, 1)
    else:
        reduce = lambda x: torch.sum(x, 1)

    total_kld = reduce(klds).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = reduce(klds).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def get_activation_name(activation):
    """
    Given a string or a `torch.nn.modules.activation`
    return the name of the activation.
    """
    if isinstance(activation, str):
        return activation

    mapper = {nn.LeakyReLU: "leaky_relu", nn.ReLU: "relu", nn.Tanh: "tanh",
              nn.Sigmoid: "sigmoid", nn.Softmax: "sigmoid"}
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))


def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain


def linear_init(layer, activation="relu"):
    """Initialize a linear layer.
    Args:
        layer (nn.Linear): parameters to initialize.
        activation (`torch.nn.modules.activation` or str, optional) activation that
            will be used on the `layer`.
    """
    x = layer.weight

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity='leaky_relu')
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity='relu')
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))


def weights_init(module):
    if isinstance(module, nn.modules.conv._ConvNd) or isinstance(module, nn.Linear):
        # TO-DO: check litterature
        linear_init(module)


def xavier_normal_init_(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        try:
            m.bias.data.zero_()
        except AttributeError:
            pass


def kaiming_normal_init_(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        try:
            m.bias.data.zero_()
        except AttributeError:
            pass
