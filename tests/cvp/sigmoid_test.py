"""Test CVP of sigmoid layer."""

import torch
from torch.nn import Sigmoid
from bpexts.cvp.sigmoid import CVPSigmoid
from .cvp_test import set_up_cvp_tests

# hyper-parameters
input_size = (10, 4, 5)
atol = 1e-7
rtol = 1e-5
num_hvp = 10


def torch_fn():
    """Create a sigmoid layer in torch."""
    return Sigmoid()


def cvp_fn():
    """Create a sigmoid layer with CVP functionality."""
    return CVPSigmoid()


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_fn,
        'CVPSigmoid',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def cvp_from_torch_fn():
    """Create CVPSigmoid from torch layer."""
    torch_layer = torch_fn()
    return CVPSigmoid.from_torch(torch_layer)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_from_torch_fn,
        'CVPSigmoidfromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
