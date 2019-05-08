"""Test HBP of MaxPool2d layer."""

import torch
from torch.nn import MaxPool2d
from bpexts.hbp.maxpool2d import HBPMaxPool2d
from .hbp_test import set_up_hbp_tests

# hyper-parameters
channels = 3
input_size = (1, channels, 4, 4)  #(1, channels, 7, 5)
atol = 5e-5
rtol = 1e-5
num_hvp = 10
kernel_size = (3, 3)
padding = 1
stride = 2
dilation = 1


def torch_fn():
    """Create a 2d max pool layer in torch."""
    return MaxPool2d(
        kernel_size, stride=stride, padding=padding, dilation=dilation)


def hbp_fn():
    """Create a 2d max pool layer with HBP functionality."""
    return HBPMaxPool2d(
        kernel_size, stride=stride, padding=padding, dilation=dilation)


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_fn,
        'HBPMaxPool2d',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def hbp_from_torch_fn():
    """Create HBPMaxPool2d from MaxPool2d."""
    torch_layer = torch_fn()
    return HBPMaxPool2d.from_torch(torch_layer)


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_from_torch_fn,
        'HBPMaxPool2dFromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
