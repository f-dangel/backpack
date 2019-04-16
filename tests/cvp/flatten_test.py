"""Test CVP of view operation."""

import torch
from torch.nn import Module
from bpexts.cvp.flatten import CVPFlatten
from bpexts.utils import Flatten
from .cvp_test import set_up_cvp_tests

# hyper-parameters
input_size = (2, 3, 4, 5)
atol = 1e-7
rtol = 1e-5
num_hvp = 10


def torch_fn():
    return Flatten()


def cvp_fn():
    return CVPFlatten()


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_fn,
        'CVPFlatten',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def cvp_from_torch_fn():
    """Create CVPFlatten from Flatten."""
    torch_layer = torch_fn()
    return CVPFlatten.from_torch(torch_layer)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_from_torch_fn,
        'CVPFlattenFromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
