"""Test CVP of Conv2d layer with trivial hyper parameters.

No padding, stride 1, dilation 1
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
atol = 1e-4
rtol = 1e-4
num_hvp = 10
kernel_size = (3, 3)
padding = 0
stride = 1
dilation = 1


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
