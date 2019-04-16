"""Test HBP of composition of ReLU and linear layer."""

from bpexts.hbp.combined_relu import HBPReLULinear
from bpexts.utils import set_seeds, ReLULinear
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
    """Create a ReLU linear layer in torch."""
    set_seeds(0)
    return ReLULinear(
        in_features=in_features, out_features=out_features, bias=bias)


def hbp_fn():
    """Create a ReLU linear layer with HBP functionality."""
    set_seeds(0)
    return HBPReLULinear(
        in_features=in_features, out_features=out_features, bias=bias)


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_fn,
        'HBPReLULinear',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
