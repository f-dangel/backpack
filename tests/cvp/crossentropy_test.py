"""Test CVP of cross-entropy loss."""

import torch
from torch.nn import CrossEntropyLoss
from bpexts.cvp.crossentropy import CVPCrossEntropyLoss
from .cvp_loss_test import set_up_cvp_loss_tests

# hyper-parameters
batch = 10
features = 20
input_size = (batch, features)
atol = 1e-8
rtol = 1e-5
num_hvp = 10


def torch_fn():
    """Create a cross-entropy layer in torch."""
    return CrossEntropyLoss()


def cvp_fn():
    """Create a cross-entropy with CVP functionality."""
    return CVPCrossEntropyLoss()


for name, test_cls in set_up_cvp_loss_tests(
        torch_fn,
        cvp_fn,
        'CVPCrossEntropyLoss',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
