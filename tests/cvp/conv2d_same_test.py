"""Test CVP of Conv2d layer with padding same."""

import torch
from bpexts.cvp.sequential import CVPConv2dSame
from bpexts.utils import set_seeds, Conv2dSame
from .cvp_test import set_up_cvp_tests

# hyper-parameters
in_channels, out_channels = 3, 2
input_size = (3, in_channels, 7, 5)
bias = True
atol = 5e-5
rtol = 5e-5
num_hvp = 10
kernel_size = (3, 3)
stride = 1
dilation = 1


def torch_fn():
    """Create a 2d convolution layer with same padding in torch."""
    set_seeds(0)
    return Conv2dSame(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        bias=bias)


def cvp_fn():
    """Create 2d convolution layer with same padding and CVP functionality."""
    set_seeds(0)
    return CVPConv2dSame(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        bias=bias)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_fn,
        'CVPConv2dSame',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def cvp_from_torch_fn():
    """Create 2d convolution with CVP from torch layer."""
    torch_layer = torch_fn()
    return CVPConv2dSame.from_torch(torch_layer)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_from_torch_fn,
        'CVPConv2dSameFromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
