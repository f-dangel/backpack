"""Test configurations to test diag_ggn.

The tests are taken from `test.extensions.secondorder.secondorder_settings`, 
but additional custom tests can be defined here by appending it to the list.
"""
from test.core.derivatives.utils import classification_targets, regression_targets
from test.extensions.automated_settings import make_simple_act_setting
from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS

import torch
from torch.nn import ELU, RNN, SELU, Flatten, Sequential

from backpack import branching
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

###############################################################################
#                         test setting: Activation Layers                     #
###############################################################################
activations = [ELU, SELU]

for act in activations:
    for bias in [True, False]:
        LOCAL_SETTINGS.append(make_simple_act_setting(act, bias=bias))


###############################################################################
#                               Branched models                               #
###############################################################################
# TODO Integrate with LOCAL_SETTINGS after integrating branching with the extensions
# DiagGGNMC, BatchDiagGGNExact
BRANCHING_SETTINGS = [
    {
        "input_fn": lambda: torch.rand(3, 10),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            # skip connection
            branching.Parallel(
                branching.ActiveIdentity(),
                torch.nn.Linear(5, 5),
            ),
            # end of skip connection
            torch.nn.Sigmoid(),
            torch.nn.Linear(5, 4),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((3,), 4),
        "id_prefix": "branching-linear",
    },
    {
        "input_fn": lambda: torch.rand(4, 2, 6, 6),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # skip connection
            branching.Parallel(
                branching.ActiveIdentity(),
                torch.nn.Sequential(
                    torch.nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1),
                ),
            ),
            # end of skip connection
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(12, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((4,), 5),
        "id_prefix": "branching-convolution",
    },
    {
        "input_fn": lambda: torch.rand(4, 3, 6, 6),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # skip connection
            branching.Parallel(
                branching.ActiveIdentity(),
                torch.nn.Sequential(
                    torch.nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1),
                    torch.nn.Sigmoid(),
                    torch.nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1),
                    branching.Parallel(
                        branching.ActiveIdentity(),
                        torch.nn.Sequential(
                            torch.nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1),
                        ),
                    ),
                ),
            ),
            # end of skip connection
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((4,), 5),
        "id_prefix": "nested-branching-convolution",
    },
]


DiagGGN_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS
