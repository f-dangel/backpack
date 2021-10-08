"""Contains tests of sub-sampling functionality."""

from torch import allclose, manual_seed, rand

from backpack.utils.subsampling import subsample


def test_subsample():
    """Test slicing operations for sub-sampling a tensor's batch axis."""
    manual_seed(0)
    tensor = rand(3, 4, 5, 6)

    # leave tensor untouched when `subsampling = None`
    assert id(subsample(tensor)) == id(tensor)
    assert allclose(subsample(tensor), tensor)

    # slice along correct dimension
    idx = [2, 0]
    assert allclose(subsample(tensor, dim=0, subsampling=idx), tensor[idx])
    assert allclose(subsample(tensor, dim=1, subsampling=idx), tensor[:, idx])
    assert allclose(subsample(tensor, dim=2, subsampling=idx), tensor[:, :, idx])
    assert allclose(subsample(tensor, dim=3, subsampling=idx), tensor[:, :, :, idx])
