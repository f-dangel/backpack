"""Model architectures used in Chen et al.: BDA-PCH (2018).

The same initialization method for the parameters was chosen.
"""

import torch.nn as nn
from bpexts.hbp.linear import HBPLinear
from bpexts.hbp.sigmoid import HBPSigmoid
from bpexts.hbp.combined_sigmoid import HBPSigmoidLinear
from bpexts.hbp.sequential import HBPSequential
from bpexts.utils import set_seeds


def original_mnist_model(seed=None):
    """FCNN architecture used by Chen et al on MNIST.

    The architecture uses the following structure:
        (784->512)->(sigmoid)->(512->128)->(sigmoid)->(128->32)->
        (sigmoid)->(32->10)

    Use Xavier initialization method for weights, set bias to 0.

    Parameters:
    -----------
    seed : int
        Random seed used for initialization process of layers,
        No seed will be set if left `None`

    Returns:
    --------
    HBPSequential:
        `PyTorch` module behaving like `nn.Sequential` with HBP
        functionality

    Examples:
    ---------
    The equivalent `PyTorch` model architecture looks as follows
    >>> import torch.nn as nn
    >>> nn.Sequential(nn.Linear(784, 512),
                      nn.Sigmoid(),
                      nn.Linear(512, 128),
                      nn.Sigmoid(),
                      nn.Linear(128, 32),
                      nn.Sigmoid(),
                      nn.Linear(32, 10))
    """
    set_seeds(seed)
    model = HBPSequential(HBPLinear(784, 512),
                          HBPSigmoidLinear(512, 128),
                          HBPSigmoidLinear(128, 32),
                          HBPSigmoidLinear(32, 10))
    xavier_init(model)
    return model


def separated_mnist_model(seed=None):
    """Original MNIST model with activations treated separately in HBP.
    
    Parameters:
    -----------
    seed : (int)
        Set seed before weight initialization, no reset if left `None`
    """
    set_seeds(seed)
    model = HBPSequential(HBPLinear(784, 512),
                          HBPSigmoid(),
                          HBPLinear(512, 128),
                          HBPSigmoid(),
                          HBPLinear(128, 32),
                          HBPSigmoid(),
                          HBPLinear(32, 10))
    xavier_init(model)
    return model


def original_cifar10_model(seed=None):
    """FCNN architecture used by Chen et al on CIFAR-10.

    The architecture uses the following neuron structure:
        3072-1024-512-256-128-64-32-16-10
    with sigmoid activation functions and linear outputs.

    Use Xavier initialization method for weights, set bias to 0.

    Parameters:
    -----------
    seed : (int)
        Random seed used for initialization process of layers,
        no reset if left `None`

    Returns:
    --------
    HBPSequential: Fully-connected deep network with described architecture
             and positive curvature Hessian backpropagation

    Examples:
    ---------
    The equivalent `PyTorch` model architecture looks as follows
    >>> import torch.nn as nn
    >>> nn.Sequential(nn.Linear(3072, 1024),
                      nn.Sigmoid(),
                      nn.Linear(1024, 512),
                      nn.Sigmoid(),
                      nn.Linear(512, 256),
                      nn.Sigmoid(),
                      nn.Linear(256, 128),
                      nn.Sigmoid(),
                      nn.Linear(128, 64),
                      nn.Sigmoid(),
                      nn.Linear(64, 32),
                      nn.Sigmoid(),
                      nn.Linear(32, 16),
                      nn.Sigmoid(),
                      nn.Linear(16, 10))
    """
    set_seeds(seed)
    model = HBPSequential(HBPLinear(3072, 1024),
                          HBPSigmoidLinear(1024, 512),
                          HBPSigmoidLinear(512, 256),
                          HBPSigmoidLinear(256, 128),
                          HBPSigmoidLinear(128, 64),
                          HBPSigmoidLinear(64, 32),
                          HBPSigmoidLinear(32, 16),
                          HBPSigmoidLinear(16, 10))
    xavier_init(model)
    return model


def separated_cifar10_model(seed=None):
    """Original CIFAR-10 model with activations treated separately in HBP.
    
    Parameters:
    -----------
    seed : (int)
        Set random seed before layer initialization, no reset if left `None`
    """
    set_seeds(seed)
    model = HBPSequential(HBPLinear(3072, 1024),
                          HBPSigmoid(),
                          HBPLinear(1024, 512),
                          HBPSigmoid(),
                          HBPLinear(512, 256),
                          HBPSigmoid(),
                          HBPLinear(256, 128),
                          HBPSigmoid(),
                          HBPLinear(128, 64),
                          HBPSigmoid(),
                          HBPLinear(64, 32),
                          HBPSigmoid(),
                          HBPLinear(32, 16),
                          HBPSigmoid(),
                          HBPLinear(16, 10))
    xavier_init(model)
    return model


def xavier_init(model):
    """Initialize weights with Xavier method, set bias to 0.

    Parameters will be modified in-place.

    Parameters:
    -----------
    model (torch.nn.Module): Net whose submodules will be initialized.
    """
    for module in model.modules():
        if hasattr(module, 'weight'):
            nn.init.xavier_normal_(module.weight)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
