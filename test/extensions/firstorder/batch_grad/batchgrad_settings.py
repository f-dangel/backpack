"""Test configurations to test batch_grad

The tests are taken from `test.extensions.firstorder.firstorder_settings`, 
but additional custom tests can be defined here by appending it to the list.
"""
from test.core.derivatives.utils import classification_targets
from test.extensions.firstorder.firstorder_settings import FIRSTORDER_SETTINGS

import torch
from torch.nn import RNN

BATCHGRAD_SETTINGS = []

SHARED_SETTINGS = FIRSTORDER_SETTINGS
LOCAL_SETTING = [
    {
        "input_fn": lambda: torch.rand(5, 8, 6),
        "module_fn": lambda: RNN(input_size=6, hidden_size=3),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((8, 3), 2),
        "axis_batch": 1,
    },
]

BATCHGRAD_SETTINGS = SHARED_SETTINGS + LOCAL_SETTING
