"""Test CVP of 2d max pool layer (non-overlapping)."""

import torch
from torch.nn import MaxPool2d
from bpexts.cvp.maxpool2d import CVPMaxPool2d
from .cvp_test import set_up_cvp_tests

# hyper-parameters
input_size = (4, 3, 7, 7)
kernel_size = (3, 3)
padding = 1
stride = (2, 2)
dilation = 1
atol = 5e-5
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
        'CVPMaxPool2dOverlap',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
