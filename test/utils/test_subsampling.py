"""Contains tests of sub-sampling functionality."""

from pytest import raises
from torch import allclose, manual_seed, rand
from torch.nn import Linear, ReLU, Sequential

from backpack.custom_module.permute import Permute
from backpack.utils.subsampling import get_batch_axis, subsample


def test_get_batch_axis():
    """Test batch axis detection."""
    # invalid argument
    with raises(ValueError):
        invalid_io_str = "dummy"
        some_module = Linear(1, 1)
        get_batch_axis(some_module, invalid_io_str)

    # Sequential with unaltered batch axis
    model = Sequential(Linear(1, 1), ReLU())
    assert get_batch_axis(model, "input0") == 0
    assert get_batch_axis(model, "output") == 0

    # Sequential with altered batch axis
    model = Sequential(Linear(1, 1), Permute(1, 0))
    assert get_batch_axis(model, "input0") == 0
    assert get_batch_axis(model, "output") == 1

    # Permute
    model = Permute(1, 3, 2, 0, batch_axis=0)
    assert get_batch_axis(model, "input0") == 0
    assert get_batch_axis(model, "output") == 3

    model = Sequential(Permute(0, 1), ReLU())
    assert get_batch_axis(model, "input0") == 0
    # expected failure due to local inspection
    batch_axis_output = 1
    assert get_batch_axis(model, "output") != batch_axis_output


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
