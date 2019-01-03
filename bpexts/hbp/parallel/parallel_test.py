"""Test splitting parameters of identical parallel layers into blocks."""

from torch import randn
from .parallel import HBPParallel
from ..linear import HBPLinear


def test_split_into_blocks():
    """Test whether block size adaption works correctly."""
    in_features = 20
    out_features_list = [3, 2, 5]
    out_features = sum(out_features_list)

    linear = HBPLinear(in_features=in_features,
                       out_features=out_features,
                       bias=True)
    input = randn(1, in_features)

    parallel = HBPParallel(linear)
    # intialize out_features_list
    parallel(input)

    assert parallel.compute_out_features_list(1) == [10]
    assert parallel.compute_out_features_list(4) == [3, 3, 2, 2]
    assert parallel.compute_out_features_list(5) == [2, 2, 2, 2, 2]
    assert parallel.compute_out_features_list(6) == [2, 2, 2, 2, 1, 1]
    assert parallel.compute_out_features_list(10) == 10 * [1]
    assert parallel.compute_out_features_list(15) == 10 * [1]

    parallel2 = HBPParallel(*[HBPLinear(in_features=in_features,
                                        out_features=out,
                                        bias=True)
                              for out in out_features_list])
    # intialize out_features_list
    parallel2(input)

    assert parallel2.compute_out_features_list(1) == [10]
    assert parallel2.compute_out_features_list(4) == [3, 3, 2, 2]
    assert parallel2.compute_out_features_list(5) == [2, 2, 2, 2, 2]
    assert parallel2.compute_out_features_list(6) == [2, 2, 2, 2, 1, 1]
    assert parallel2.compute_out_features_list(10) == 10 * [1]
    assert parallel2.compute_out_features_list(15) == 10 * [1]
