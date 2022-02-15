"""Tests for ``backpack.custom_module.slicing.Slicing``."""

from test.core.derivatives.utils import get_available_devices
from typing import Dict

import torch
from pytest import mark
from torch import allclose, manual_seed, rand

from backpack.custom_module.slicing import Slicing

CONFIGURATIONS = [
    {
        "input_fn": lambda: rand(2, 3, 4, 5),
        "slice_info": (0,),
        "seed": 0,
    },
    {
        "input_fn": lambda: rand(2, 3, 4, 5),
        "slice_info": (slice(None), 0, slice(0, 2), slice(1, 5, 2)),
        "seed": 1,
    },
    {
        "input_fn": lambda: rand(2, 3, 4, 5),
        "slice_info": (slice(1, 2), 0, slice(None), slice(1, 5, 2)),
        "seed": 1,
    },
]
DEVICES = get_available_devices()


@mark.parametrize("device", DEVICES, ids=str)
@mark.parametrize("config", CONFIGURATIONS, ids=str)
def test_slicing_forward(config: Dict, device: torch.device):
    """Test forward pass of the custom slicing module.

    Args:
        config: Dictionary specifying the test case.
        device: Device to execute the test on.
    """
    manual_seed(config["seed"])

    input = config["input_fn"]().to(device)
    slice_info = config["slice_info"]

    layer = Slicing(slice_info).to(device)

    assert allclose(layer(input), input[slice_info])
