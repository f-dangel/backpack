"""Test HBP of composition of Sigmoid and linear layer."""

from bpexts.hbp.combined_sigmoid import HBPSigmoidLinear
from bpexts.utils import set_seeds, SigmoidLinear
from .hbp_test import set_up_hbp_tests

# hyper-parameters
in_features = 50
out_features = 10
bias = True
input_size = (1, in_features)
atol = 5e-5
rtol = 1e-5
num_hvp = 10


def torch_fn():
    """Create a Sigmoid linear layer in torch."""
    set_seeds(0)
    return SigmoidLinear(
        in_features=in_features, out_features=out_features, bias=bias)


def hbp_fn():
    """Create a Sigmoid linear layer with HBP functionality."""
    set_seeds(0)
    return HBPSigmoidLinear(
        in_features=in_features, out_features=out_features, bias=bias)


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_fn,
        'HBPSigmoidLinear',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def hbp_from_torch_fn():
    """Create HBPReLULinear from ReLULinear."""
    torch_layer = torch_fn()
    return HBPSigmoidLinear.from_torch(torch_layer)


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_from_torch_fn,
        'HBPSigmoidLinearFromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
