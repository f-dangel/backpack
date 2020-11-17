"""Test configurations to test diag_ggn

The tests are taken from `test.extensions.secondorder.secondorder_settings`, 
but additional custom tests can be defined here by appending it to the list.
"""

from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS
import torch
from torch.nn import ReLU, Sigmoid, Tanh, LogSigmoid, LeakyReLU, ELU, SELU

from test.extensions.automated_settings import make_simple_cnn_setting
from test.core.derivatives.utils import classification_targets

DiagGGN_SETTINGS = []

SHARED_SETTINGS = SECONDORDER_SETTINGS
LOCAL_SETTINGS = [
    {
        "input_fn": lambda: torch.rand(3, 3, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv1d(3, 2, 2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(12, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 3, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv1d(3, 2, 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(12, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 3, 8),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv1d(
                3, 6, 2, stride=4, padding=2, padding_mode="zeros", dilation=3
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(18, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 3, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv1d(3, 2, 2, padding=2, dilation=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(10, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 2, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv1d(2, 3, 2, padding=0, dilation=2, groups=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(15, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 3, 2, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv3d(3, 2, 2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(72, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 3, 2, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv3d(3, 2, 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(72, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 3, 4, 8, 8),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv3d(
                3, 6, 2, padding=2, stride=4, dilation=3, padding_mode="zeros"
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(108, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 3, 2, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv3d(3, 2, 2, dilation=1, padding=2, stride=3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 2, 3, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv3d(2, 3, 2, dilation=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(75, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
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
        "input_fn": lambda: torch.rand(3, 3, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.ConvTranspose1d(3, 2, 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 3, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.ConvTranspose1d(3, 2, 2, padding=2, dilation=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(20, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 2, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.ConvTranspose1d(2, 3, 2, padding=0, dilation=5, stride=3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(72, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
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
        "input_fn": lambda: torch.rand(3, 3, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3, 2, 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 2, 9, 9),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.ConvTranspose2d(2, 4, 2, padding=0, dilation=2, groups=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(484, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(2, 3, 2, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.ConvTranspose3d(3, 2, 2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(384, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((2,), 5),
    },
    {
        "input_fn": lambda: torch.rand(2, 3, 2, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.ConvTranspose3d(3, 2, 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(384, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((2,), 5),
    },
    {
        "input_fn": lambda: torch.rand(2, 3, 5, 5, 5),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.ConvTranspose3d(3, 2, 2, padding=2, dilation=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(686, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((2,), 5),
    },
]

###############################################################################
#                         test setting: Activation Layers                     #
###############################################################################
activations = [ReLU, Sigmoid, Tanh, LeakyReLU, LogSigmoid, ELU, SELU]

for act in activations:
    for bias in [True, False]:
        LOCAL_SETTINGS.append(make_simple_cnn_setting(act, bias=bias))

DiagGGN_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS
