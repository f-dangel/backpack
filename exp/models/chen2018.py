"""Model architectures used in Chen et al.: BDA-PCH (2018).

The same initialization method for the parameters is chosen.
"""

import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, Sigmoid


def mnist_model():
    """FCNN architecture used by Chen et al on MNIST.

    The architecture uses the following structure:
        (784->512)->(sigmoid)->(512->128)->(sigmoid)->(128->32)->
        (sigmoid)->(32->10)

    Use Xavier initialization method for weights, set bias to 0.
    """
    model = Sequential(
        Linear(784, 512), Sigmoid(), Linear(512, 128), Sigmoid(),
        Linear(128, 32), Sigmoid(), Linear(32, 10))
    xavier_init(model)
    return model


def cifar10_model():
    """FCNN architecture used by Chen et al on CIFAR-10.

    The architecture uses the following neuron structure:
        3072-1024-512-256-128-64-32-16-10
    with sigmoid activation functions and linear outputs.

    Use Xavier initialization method for weights, set bias to 0.
    """
    model = Sequential(
        Linear(3072, 1024), Sigmoid(), Linear(1024, 512), Sigmoid(),
        Linear(512, 256), Sigmoid(), Linear(256, 128), Sigmoid(),
        Linear(128, 64), Sigmoid(), Linear(64, 32), Sigmoid(), Linear(32, 16),
        Sigmoid(), Linear(16, 10))
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
