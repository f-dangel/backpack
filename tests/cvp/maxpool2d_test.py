"""Test CVP of 2d max pool layer."""

import torch
from torch.nn import MaxPool2d
from bpexts.cvp.maxpool2d import CVPMaxPool2d
from .cvp_test import set_up_cvp_tests

# hyper-parameters
input_size = (1, 3, 8, 8)
kernel_size = (4, 4)
padding = 2
stride = None
dilation = 1
atol = 1e-5
rtol = 1e-5
num_hvp = 10


def torch_fn():
    """Create a 2d max pool layer in torch."""
    return MaxPool2d(
        kernel_size, padding=padding, dilation=dilation, stride=stride)


def cvp_fn():
    """Create a 2d max pool layer with CVP functionality."""
    return CVPMaxPool2d(
        kernel_size, padding=padding, dilation=dilation, stride=stride)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_fn,
        'CVPMaxPool2d',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
