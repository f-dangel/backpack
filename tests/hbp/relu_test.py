"""Test HBP of sigmoid layer."""

from torch.nn import ReLU
from bpexts.hbp.relu import HBPReLU
from .hbp_test import set_up_hbp_tests

# hyper-parameters
in_features = 30
bias = True
input_size = (1, in_features)
atol = 5e-6
rtol = 1e-5
num_hvp = 10


def torch_fn():
    """Create a ReLU layer in torch."""
    return ReLU()


def hbp_fn():
    """Create a ReLU layer with HBP functionality."""
    return HBPReLU()


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_fn,
        'HBPReLU',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def hbp_from_torch_fn():
    """Create HBPReLU from ReLU."""
    torch_layer = torch_fn()
    return HBPReLU.from_torch(torch_layer)


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_from_torch_fn,
        'HBPReLUFromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
