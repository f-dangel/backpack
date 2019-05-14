"""Test CVP of tanh layer."""

import torch
from torch.nn import Tanh
from bpexts.cvp.tanh import CVPTanh
from .cvp_test import set_up_cvp_tests

# hyper-parameters
input_size = (10, 4, 5)
atol = 1e-7
rtol = 1e-5
num_hvp = 10


def torch_fn():
    """Create a tanh layer in torch."""
    return Tanh()


def cvp_fn():
    """Create a tanh layer with CVP functionality."""
    return CVPTanh()


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_fn,
        'CVPTanh',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def cvp_from_torch_fn():
    """Create CVPTanh from torch layer."""
    torch_layer = torch_fn()
    return CVPTanh.from_torch(torch_layer)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_from_torch_fn,
        'CVPTanhFromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
