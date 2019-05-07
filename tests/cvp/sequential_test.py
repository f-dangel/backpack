"""Test CVP for a sequence of layers."""

import torch
import numpy
# torch layers
from torch.nn import (Linear, Sigmoid, ReLU, Conv2d, MaxPool2d, Sequential)
# CVP layers
from bpexts.cvp.linear import CVPLinear
from bpexts.cvp.sigmoid import CVPSigmoid
from bpexts.cvp.relu import CVPReLU
from bpexts.cvp.conv2d import CVPConv2d
from bpexts.cvp.maxpool2d import CVPMaxPool2d
from bpexts.cvp.flatten import CVPFlatten
from bpexts.cvp.sequential import CVPSequential
from bpexts.utils import set_seeds, Flatten
from bpexts.hbp.conv2d import HBPConv2d
from .cvp_test import set_up_cvp_tests

# number of tests and accuracy
atol = 1e-6
rtol = 1e-5
num_hvp = 10

# convolution parameters
in_channels = 3
input_size = (4, in_channels, 6, 4)
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
out1_numel = numpy.prod(out1_shape)
# output of max pooling
out2_shape = HBPConv2d.output_shape(
    out1_shape,
    out_channels,
    kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation)
out2_features = numpy.prod(out2_shape[1:])
# output of linear layer
out3_features = out2_features // 2 + 1
# final output
out4_features = 10


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
            bias=True), ReLU(),
        MaxPool2d(
            kernel_size, stride=stride, padding=padding, dilation=dilation),
        Flatten(), Linear(out2_features, out3_features, bias=False), Sigmoid(),
        Linear(out3_features, out4_features, bias=True))


def cvp_fn():
    """Create sequence of layers in torch."""
    set_seeds(0)
    return CVPSequential(
        CVPConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True), CVPReLU(),
        CVPMaxPool2d(
            kernel_size, stride=stride, padding=padding, dilation=dilation),
        CVPFlatten(), CVPLinear(out2_features, out3_features, bias=False),
        CVPSigmoid(), CVPLinear(out3_features, out4_features, bias=True))


print(torch_fn())
print(cvp_fn())

for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_fn,
        'CVPSequential',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def cvp_from_torch_fn():
    """Create CVPSequential from torch layer."""
    torch_layer = torch_fn()
    return CVPSequential.from_torch(torch_layer)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_from_torch_fn,
        'CVPSequentialFromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
