"""Contains tests for BackPACK's custom ``Permute`` module."""

from pytest import raises

from backpack.custom_module.permute import Permute


def test_get_batch_axis():
    """Test the Permute module's batch axis detection."""
    # invalid argument
    with raises(ValueError):
        invalid_io_str = "dummy"
        Permute().get_batch_axis(invalid_io_str)

    # batch axis unaffected by forward pass
    assert Permute(0, 2, 1).get_batch_axis("input0") == 0
    assert Permute(0, 2, 1).get_batch_axis("output") == 0

    # batch axis first, affected by forward pass
    assert Permute(1, 2, 0).get_batch_axis("input0") == 0
    assert Permute(1, 2, 0).get_batch_axis("output") == 2

    # batch axis second, affected by forward pass
    assert Permute(1, 2, 0, batch_axis=1).get_batch_axis("input0") == 1
    assert Permute(1, 2, 0, batch_axis=1).get_batch_axis("output") == 0
