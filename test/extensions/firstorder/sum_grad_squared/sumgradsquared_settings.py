"""Test configurations to test sum_grad_square

The tests are taken from `test.extensions.firstorder.firstorder_settings`, 
but additional custom tests can be defined here by appending it to the list.
"""
import torch

from test.extensions.firstorder.firstorder_settings import FIRSTORDER_SETTINGS
from test.core.derivatives.utils import classification_targets

SHARED_SETTINGS = FIRSTORDER_SETTINGS
LOCAL_SETTINGS = [
    {
        "input_fn": lambda: torch.rand(3, 3, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.ConvTranspose1d(3, 2, 2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 2, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.ConvTranspose2d(2, 2, 2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 3, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3, 2, 2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 2, 2, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.ConvTranspose3d(2, 2, 2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(384, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 3, 2, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.ConvTranspose3d(3, 2, 2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(384, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((3,), 5),
    },
]

SUMGRADSQUARED_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS
