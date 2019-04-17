"""Test CVP of MaxPool2d layer with padding same."""

import torch
from bpexts.cvp.sequential import CVPMaxPool2dSame
from bpexts.utils import set_seeds, MaxPool2dSame
from .cvp_test import set_up_cvp_tests

# hyper-parameters
input_size = (4, 3, 8, 8)
kernel_size = (3, 3)
stride = (1, 1)
dilation = 1
atol = 1e-5
rtol = 1e-5
num_hvp = 10


def torch_fn():
    """Create a 2d maxpool layer with same padding in torch."""
    return MaxPool2dSame(kernel_size, stride=stride, dilation=dilation)


def cvp_fn():
    """Create 2d maxpool layer with same padding and CVP functionality."""
    return CVPMaxPool2dSame(kernel_size, stride=stride, dilation=dilation)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_fn,
        'CVPMaxPool2dSame',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def cvp_from_torch_fn():
    """Create 2d maxpool layer with CVP from torch layer."""
    torch_layer = torch_fn()
    return CVPMaxPool2dSame.from_torch(torch_layer)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_from_torch_fn,
        'CVPMaxPool2dSameFromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
