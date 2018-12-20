"""Test conversion HBPLinear to parallel series and splitting."""

from torch import randn
from ..linear import HBPLinear
from .identical import HBPParallelIdentical
from ...utils import (torch_allclose,
                      set_seeds)


in_features = 20
out_features_list = [2, 3, 4]
out_features_list2 = [1, 8]
num_layers = len(out_features_list)
input = randn(1, in_features)


def random_input():
    """Return random input copy."""
    return input.clone()


def test_forward_pass_hbp_linear():
    """Test whether single/split modules are consistent in forward."""
    linear = HBPLinear(in_features=in_features,
                       out_features=sum(out_features_list),
                       bias=True)
    x = random_input()
    parallel = HBPParallelIdentical.from_module(linear)
    assert torch_allclose(linear(x), parallel(x))

    parallel2 = parallel.split(out_features_list)
    assert torch_allclose(linear(x), parallel2(x))

    parallel3 = parallel2.unite()

    assert torch_allclose(linear.bias, parallel3.get_submodule(0).bias)
    assert torch_allclose(linear.weight, parallel3.get_submodule(0).weight)

    assert torch_allclose(linear(x), parallel3(x))

    parallel4 = parallel2.split(out_features_list2)
    assert torch_allclose(linear(x), parallel4(x))
