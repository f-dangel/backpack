"""Test CVP of 2d zero padding operation."""

from bpexts.cvp.padding import CVPZeroPad2d
from torch.nn import ZeroPad2d
from .cvp_test import set_up_cvp_tests

# hyper-parameters
input_size = (2, 3, 4, 5)
padding = (2, 1, 3, 4)
atol = 1e-7
rtol = 1e-5
num_hvp = 10


def torch_fn():
    return ZeroPad2d(padding)


def cvp_fn():
    return CVPZeroPad2d(padding)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_fn,
        'CVPZeroPad2d',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def cvp_from_torch_fn():
    """Create CVPZeroPad2d from ZeroPad2d."""
    torch_layer = torch_fn()
    return CVPZeroPad2d.from_torch(torch_layer)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_from_torch_fn,
        'CVPZeroPad2dFromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
