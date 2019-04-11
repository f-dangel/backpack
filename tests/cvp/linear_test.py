"""Test CVP of linear layer."""

import torch
from torch.nn import Linear
from bpexts.cvp.linear import CVPLinear
from bpexts.utils import set_seeds
from .cvp_test import set_up_cvp_tests

# hyper-parameters
in_features = 20
out_features = 10
batch = 3
bias = True
input_size = (batch, in_features)
atol = 1e-5
rtol = 1e-5
num_hvp = 10


def torch_fn():
    """Create a linear layer in torch."""
    set_seeds(0)
    return Linear(
        in_features=in_features, out_features=out_features, bias=bias)


def cvp_fn():
    """Create a linear layer with HBP functionality."""
    set_seeds(0)
    return CVPLinear(
        in_features=in_features, out_features=out_features, bias=bias)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_fn,
        'CVPLinear',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
