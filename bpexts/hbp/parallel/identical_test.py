"""Test splitting parameters of identical parallel layers into blocks."""

from torch import randn
from ..linear import HBPLinear
from .identical import HBPParallelIdentical
from ...utils import (torch_allclose,
                      set_seeds)

in_features = 20
out_features_list = [3, 2, 5]
out_features = sum(out_features_list)


def test_splitting_into_blocks():
    """Test whether block size adaption works correctly."""
    linear = HBPLinear(in_features=in_features,
                       out_features=out_features,
                       bias=True)

    parallel = HBPParallelIdentical.from_module(linear)
    assert parallel.total_out_features() == out_features
    assert parallel.compute_out_features_list(4) == [3, 3, 2, 2]
    assert parallel.compute_out_features_list(5) == [2, 2, 2, 2, 2]
    assert parallel.compute_out_features_list(6) == [2, 2, 2, 2, 1, 1]
    assert parallel.compute_out_features_list(10) == 10 * [1]
    assert parallel.compute_out_features_list(15) == 10 * [1]

    parallel2 = parallel.split(out_features_list)
    assert parallel2.total_out_features() == out_features
    assert parallel2.compute_out_features_list(4) == [3, 3, 2, 2]
    assert parallel2.compute_out_features_list(5) == [2, 2, 2, 2, 2]
    assert parallel2.compute_out_features_list(6) == [2, 2, 2, 2, 1, 1]
    assert parallel2.compute_out_features_list(10) == 10 * [1]
    assert parallel2.compute_out_features_list(15) == 10 * [1]
