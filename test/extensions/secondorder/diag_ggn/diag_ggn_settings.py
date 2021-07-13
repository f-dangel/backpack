"""Test cases for BackPACK extensions for the GGN diagonal.

Includes
- ``DiagGGNExact``
- ``DiagGGNMC``
- ``BatchDiagGGNExact``
- ``BatchDiagGGNMC``

Shared settings are taken from `test.extensions.secondorder.secondorder_settings`.
Additional local cases can be defined here through ``LOCAL_SETTINGS``.
"""
from test.core.derivatives.utils import regression_targets
from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS

import torch
from torch.nn import RNN, Flatten, Sequential

from backpack.custom_module.permute import Permute
from backpack.custom_module.reduce_tuple import ReduceTuple

SHARED_SETTINGS = SECONDORDER_SETTINGS
LOCAL_SETTINGS = [
    # RNN settings
    {
        "input_fn": lambda: torch.rand(8, 5, 6),
        "module_fn": lambda: Sequential(
            Permute(1, 0, 2),
            RNN(input_size=6, hidden_size=3),
            ReduceTuple(index=0),
            Permute(1, 2, 0),
            Flatten(),
        ),
        "loss_function_fn": lambda: torch.nn.MSELoss(),
        "target_fn": lambda: regression_targets((8, 3 * 5)),
    },
]
DiagGGN_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS
