"""Test CVP of ReLU layer."""

import torch
from torch.nn import ReLU
from bpexts.cvp.relu import CVPReLU
from .cvp_test import set_up_cvp_tests

# hyper-parameters
input_size = (8, 4, 2, 3)
atol = 1e-7
rtol = 1e-5
num_hvp = 10


def torch_fn():
    """Create a sigmoid layer in torch."""
    return ReLU()


def cvp_fn():
    """Create a sigmoid layer with CVP functionality."""
    return CVPReLU()


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_fn,
        'CVPReLU',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
