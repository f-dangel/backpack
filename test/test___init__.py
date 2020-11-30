"""Tests for `backpack.__init__.py`."""

import contextlib
from test import pytorch_current_memory_usage
from test.core.derivatives.utils import classification_targets, get_available_devices

import pytest
import torch

from backpack import extend, no_io

DEVICES = get_available_devices()
DEVICES_ID = [str(dev) for dev in DEVICES]


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


def memory_leak(use_no_io, device):
    """Reproduce memory leak due to forward pass through a model with non-freed IO.

    Raises:
        RuntimeError if a memory leak is detected (the allocated memory exceeds the
            specified threshold).
    """
    memory_init = pytorch_current_memory_usage()

    torch.manual_seed(0)

    # MNIST dummy
    B = 256
    X = torch.rand(B, 1, 28, 28).to(device)
    y = classification_targets((B,), 10).to(device)

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 10),
    ).to(device)
    model = extend(model)

    # threshold to detect a memory_leak
    steps = 50
    memory_leak_threshold_mb = 1
    memory_leak_threshold = memory_leak_threshold_mb * 2 ** 20

    context = no_io if use_no_io else contextlib.nullcontext

    for _ in range(steps):
        lossfunc = torch.nn.CrossEntropyLoss().to(device)
        lossfunc = extend(lossfunc)

        loss = lossfunc(model(X), y)
        loss.backward()

        with context():
            loss = lossfunc(model(X), y)  # this is what kills it

        memory = pytorch_current_memory_usage()
        if memory - memory_init > memory_leak_threshold:
            raise RuntimeError(
                f"Memory leak detected: use_no_io={use_no_io}, device={device}"
            )


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
def test_memory_leak(device):
    with pytest.raises(RuntimeError):
        memory_leak(use_no_io=False, device=device)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
def test_no_io_resolves_memory_leak(device):
    memory_leak(use_no_io=True, device=device)


def memory_after_forward(use_no_io, device):
    """Return memory consumed by the forward pass of an extended model."""
    memory_init = pytorch_current_memory_usage()

    torch.manual_seed(0)

    # MNIST dummy
    B = 256
    X = torch.rand(B, 1, 28, 28).to(device)
    y = classification_targets((B,), 10).to(device)

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 10),
    ).to(device)
    model = extend(model)

    lossfunc = torch.nn.CrossEntropyLoss().to(device)
    lossfunc = extend(lossfunc)

    context = no_io if use_no_io else contextlib.nullcontext

    with context():
        lossfunc(model(X), y)

    return pytorch_current_memory_usage() - memory_init


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
def test_no_io_save_memory(device):
    """Verify that ``no_io`` requires less memory."""
    with_io = memory_after_forward(use_no_io=False, device=device)
    without_io = memory_after_forward(use_no_io=True, device=device)

    assert with_io > without_io, f"With IO: {with_io}, without_io: {without_io}"
