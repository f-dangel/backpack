"""Test HBP of linear layer."""

import torch
from torch.nn import Linear
from bpexts.hbp.linear import HBPLinear
from bpexts.utils import set_seeds
from hbp_test import hbp_test

# hyper-parameters
in_features = 100
out_features = 10
bias = True
input_size = (1, in_features)
atol = 1e-4
num_hvp = 100


def torch_fn():
    """Create a linear layer in torch."""
    set_seeds(0)
    return Linear(
        in_features=in_features, out_features=out_features, bias=bias)


def hbp_fn():
    """Create a linear layer with HBP functionality."""
    set_seeds(0)
    return HBPLinear(
        in_features=in_features, out_features=out_features, bias=bias)


class HBPLinearCPUTest(
        hbp_test(
            torch_fn,
            hbp_fn,
            input_size,
            device=torch.device('cpu'),
            atol=atol,
            num_hvp=num_hvp)):
    """Compare torch Linear and HBPLinear on CPU."""
    pass


if torch.cuda.is_available():
    device = torch.device('cpu')

    class HBPLinearGPUTest(
            hbp_test(
                torch_fn,
                hbp_fn,
                input_size,
                device=torch.device('cuda:0'),
                atol=atol,
                num_hvp=num_hvp)):
        """Compare torch Linear and HBPLinear on GPU."""
        pass
