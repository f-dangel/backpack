"""Test configurations to test batch_grad.

The tests are taken from `test.extensions.firstorder.firstorder_settings`, 
but additional custom tests can be defined here by appending it to the list.
"""
from test.core.derivatives.utils import classification_targets, regression_targets
from test.extensions.firstorder.firstorder_settings import FIRSTORDER_SETTINGS

import torch
from torch.nn import RNN, Flatten, Sequential

from backpack.custom_module.permute import Permute
from backpack.custom_module.reduce_tuple import ReduceTuple

BATCHGRAD_SETTINGS = []

SHARED_SETTINGS = FIRSTORDER_SETTINGS
LOCAL_SETTING = [
    {
        "input_fn": lambda: torch.rand(8, 5, 6),
        "module_fn": lambda: Sequential(
            Permute(dims=[1, 0, 2]),
            RNN(input_size=6, hidden_size=3),
            ReduceTuple(index=0),
            Permute(dims=[1, 2, 0]),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((8, 5), 3),
    },
    {
        "input_fn": lambda: torch.rand(8, 5, 6),
        "module_fn": lambda: Sequential(
            Permute(dims=[1, 0, 2]),
            RNN(input_size=6, hidden_size=3),
            ReduceTuple(index=0),
            Permute(dims=[1, 2, 0]),
            Flatten(),
        ),
        "loss_function_fn": lambda: torch.nn.MSELoss(),
        "target_fn": lambda: regression_targets((8, 3 * 5)),
    },
]

BATCHGRAD_SETTINGS = SHARED_SETTINGS + LOCAL_SETTING
