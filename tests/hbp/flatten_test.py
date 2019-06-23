"""Test HBP of flatten operation."""

from bpexts.hbp.flatten import HBPFlatten
from bpexts.utils import Flatten
from .hbp_test import set_up_hbp_tests

# hyper-parameters
input_size = (1, 3, 4, 5)
atol = 1e-7
rtol = 1e-5
num_hvp = 10


def torch_fn():
    return Flatten()


def cvp_fn():
    return HBPFlatten()


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        cvp_fn,
        'HBPFlatten',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def hbp_from_torch_fn():
    """Create HBPFlatten from Flatten."""
    torch_layer = torch_fn()
    return HBPFlatten.from_torch(torch_layer)


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_from_torch_fn,
        'HBPFlattenFromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
