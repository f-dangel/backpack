"""Tests for sub-sampling utilities."""

from pytest import raises
from torch.nn import LSTM, RNN, Linear, Sequential

from backpack.utils.subsampling import get_batch_axis


def test_get_batch_axis():
    """Check batch axis detection."""
    # RNNs
    assert get_batch_axis(RNN(input_size=2, hidden_size=2, batch_first=True)) == 0
    assert get_batch_axis(RNN(input_size=2, hidden_size=2, batch_first=False)) == 1
    assert get_batch_axis(LSTM(input_size=2, hidden_size=2, batch_first=True)) == 0
    assert get_batch_axis(LSTM(input_size=2, hidden_size=2, batch_first=False)) == 1

    # module
    assert get_batch_axis(Linear(in_features=3, out_features=2)) == 0
    # inside a container
    assert get_batch_axis(Sequential(Linear(in_features=3, out_features=2))) == 0

    with raises(ValueError):  # cannot be inferred, depends on submodules
        get_batch_axis(Sequential())
    with raises(ValueError):  # inconsistent
        get_batch_axis(
            Sequential(
                RNN(input_size=2, hidden_size=2, batch_first=True),
                RNN(input_size=2, hidden_size=2, batch_first=False),
            )
        )
