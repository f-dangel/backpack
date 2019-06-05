"""Test HBP for a sequence of layers."""

import torch
import numpy
# torch layers
from torch.nn import (Linear, Sigmoid, ReLU, Conv2d, MaxPool2d, Sequential)
# CVP layers
from bpexts.hbp.linear import HBPLinear
from bpexts.hbp.sigmoid import HBPSigmoid
from bpexts.hbp.relu import HBPReLU
from bpexts.hbp.conv2d import HBPConv2d
from bpexts.hbp.maxpool2d import HBPMaxPool2d
from bpexts.hbp.flatten import HBPFlatten
from bpexts.hbp.sequential import HBPSequential
from bpexts.utils import set_seeds, Flatten
from .hbp_test import set_up_hbp_tests

# number of tests and accuracy
atol = 5e-6
rtol = 1e-5
num_hvp = 10

# convolution parameters
in_channels = 3
input_size = (1, in_channels, 8, 6)
out_channels = 4
kernel_size = (3, 3)
stride = (1, 1)
padding = (1, 1)
dilation = 1

# maxpool2d parameters
pool_kernel = 2
pool_padding = 1

# output size
out1 = 80
out2 = out1 // 2
out3 = 10


def torch_fn():
    """Create sequence of layers in torch."""
    set_seeds(0)
    return Sequential(
        Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True), ReLU(), MaxPool2d(pool_kernel, padding=pool_padding),
        Flatten(), Linear(out1, out2, bias=False), Sigmoid(),
        Linear(out2, out3, bias=True))


def hbp_fn():
    """Create sequence of layers in HBP."""
    set_seeds(0)
    return HBPSequential(
        HBPConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True), HBPReLU(),
        HBPMaxPool2d(pool_kernel, padding=pool_padding), HBPFlatten(),
        HBPLinear(out1, out2, bias=False), HBPSigmoid(),
        HBPLinear(out2, out3, bias=True))


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_fn,
        'HBPSequential',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def hbp_from_torch_fn():
    """Create HBPSequential from torch layer."""
    torch_layer = torch_fn()
    return HBPSequential.from_torch(torch_layer)


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_from_torch_fn,
        'HBPSequentialFromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls

print(torch_fn())
print(hbp_fn())
print(hbp_from_torch_fn())
