"""Tests for `backpack.__init__.py`."""

import pytest
import torch

from backpack import extend, no_io


def test_no_io():
    """Check IO is not tracked."""
    torch.manual_seed(0)

    input = torch.rand(3, 5)
    module = torch.nn.Linear(5, 2)
    extend(module)

    with no_io():
        module(input)
        assert not hasattr(module, "input0")
        assert not hasattr(module, "output")

    module(input)
    assert hasattr(module, "input0")
    assert hasattr(module, "output")


def test_no_io_should_store_io():
    """Check IO tracking is disabled by ``no_io`` and nesting ``no_io``."""
    assert no_io.should_store_io()

    with no_io():
        assert not no_io.should_store_io()

        with no_io():
            assert not no_io.should_store_io()

        assert not no_io.should_store_io()

    assert no_io.should_store_io()
