"""Test HBP of tanh layer."""

import torch
from torch.nn import Tanh
from bpexts.hbp.tanh import HBPTanh
from .hbp_test import set_up_hbp_tests

# hyper-parameters
in_features = 30
bias = True
input_size = (1, in_features)
atol = 5e-6
rtol = 1e-5
num_hvp = 1000


def torch_fn():
    """Create a tanh layer in torch."""
    return Tanh()


def hbp_fn():
    """Create a tanh layer with HBP functionality."""
    return HBPTanh()


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_fn,
        'HBPTanh',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def hbp_from_torch_fn():
    """Create HBPTanh from Tanh."""
    torch_layer = torch_fn()
    return HBPTanh.from_torch(torch_layer)


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_from_torch_fn,
        'HBPTanhFromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
