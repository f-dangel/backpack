"""Tests for ``backpack.custom_module.pad.Pad``."""

from test.core.derivatives.utils import get_available_devices
from typing import Dict

import torch
from pytest import mark
from torch import allclose, manual_seed, rand
from torch.nn import functional as F

from backpack.custom_module.pad import Pad

CONFIGURATIONS = [
    {
        "input_fn": lambda: rand(2, 3, 4, 5),
        "pad": (1, 3),
        "mode": "constant",
        "value": 0.0,
        "seed": 1,
    },
    {
        "input_fn": lambda: rand(2, 3, 4, 5),
        "pad": (4, 0, 1, 3),
        "mode": "constant",
        "value": 1.0,
        "seed": 0,
    },
    {
        "input_fn": lambda: rand(2, 3, 4, 5),
        "pad": (1, 5, 4, 0, 1, 3),
        "mode": "constant",
        "value": 2.0,
        "seed": 2,
    },
]
DEVICES = get_available_devices()


@mark.parametrize("device", DEVICES, ids=str)
@mark.parametrize("config", CONFIGURATIONS, ids=str)
def test_pad_forward(config: Dict, device: torch.device):
    """Test forward pass of the custom padding module.

    Args:
        config: Dictionary specifying the test case.
        device: Device to execute the test on
    """
    manual_seed(config["seed"])

    input = config["input_fn"]().to(device)
    pad, mode, value = config["pad"], config["mode"], config["value"]

    layer = Pad(pad, mode=mode, value=value).to(device)

    assert allclose(layer(input), F.pad(input, pad, mode=mode, value=value))
