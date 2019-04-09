"""Test HBP of sigmoid layer."""

import torch
from torch.nn import Sigmoid
from bpexts.hbp.sigmoid import HBPSigmoid
from bpexts.hbp.loss import batch_summed_hessian
from hbp_test import set_up_hbp_tests

# hyper-parameters
in_features = 30
bias = True
input_size = (1, in_features)
atol = 1e-6
rtol = 1e-5
num_hvp = 10


def torch_fn():
    """Create a sigmoid layer in torch."""
    return Sigmoid()


def hbp_fn():
    """Create a sigmoid layer with HBP functionality."""
    return HBPSigmoid()


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_fn,
        'HBPSigmoid',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
