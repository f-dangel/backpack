"""Contains tests of sub-sampling functionality."""

from pytest import raises
from torch.nn import Linear, ReLU, Sequential

from backpack.custom_module.permute import Permute
from backpack.utils.subsampling import get_batch_axis


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
