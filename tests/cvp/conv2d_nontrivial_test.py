"""Test CVP of Conv2d layer with non-trivial hyper parameters.

Nontrivial padding, stride, dilation, kernel size
"""

import torch
from torch.nn import Conv2d
from bpexts.cvp.conv2d import CVPConv2d
from bpexts.utils import set_seeds
from .cvp_test import set_up_cvp_tests

# hyper-parameters
in_channels, out_channels = 3, 2
input_size = (3, in_channels, 7, 5)
bias = True
atol = 5e-5
rtol = 1e-5
num_hvp = 10
kernel_size = (3, 2)
padding = (3, 2)
stride = (1, 2)
dilation = (2, 2)


def torch_fn():
    """Create a 2d convolution layer in torch."""
    set_seeds(0)
    return Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias)


def cvp_fn():
    """Create a 2d convolution layer with CVP functionality."""
    set_seeds(0)
    return CVPConv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_fn,
        'CVPConv2d',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def cvp_from_torch_fn():
    """Create CVPConv2d from Conv2d."""
    torch_layer = torch_fn()
    return CVPConv2d.from_torch(torch_layer)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_from_torch_fn,
        'CVPConv2dFromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
