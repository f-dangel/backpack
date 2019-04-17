"""Test HBP for a sequence of layers."""

import torch
import numpy
# torch layers
from torch.nn import (Linear, Sigmoid, ReLU, Conv2d, Sequential)
# CVP layers
from bpexts.hbp.linear import HBPLinear
from bpexts.hbp.sigmoid import HBPSigmoid
from bpexts.hbp.relu import HBPReLU
from bpexts.hbp.conv2d import HBPConv2d
from bpexts.hbp.flatten import HBPFlatten
from bpexts.hbp.sequential import HBPSequential
from bpexts.utils import set_seeds, Flatten
from .hbp_test import set_up_hbp_tests

# number of tests and accuracy
atol = 1e-6
rtol = 1e-5
num_hvp = 10

# convolution parameters
in_channels = 3
input_size = (1, in_channels, 6, 4)
out_channels = 4
kernel_size = (3, 3)
stride = (1, 1)
padding = (1, 1)
dilation = 1
# output of convolution
out1_shape = HBPConv2d.output_shape(
    input_size,
    out_channels,
    kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation)
out1_features = numpy.prod(out1_shape)
# output of linear layer
out2_features = out1_features // 2 + 1
# final output
out3_features = 10


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
            bias=True), ReLU(), Flatten(),
        Linear(out1_features, out2_features, bias=False), Sigmoid(),
        Linear(out2_features, out3_features, bias=True))


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
            bias=True), HBPReLU(), HBPFlatten(),
        HBPLinear(out1_features, out2_features, bias=False), HBPSigmoid(),
        HBPLinear(out2_features, out3_features, bias=True))


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
