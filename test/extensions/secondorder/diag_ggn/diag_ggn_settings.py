"""Test cases for BackPACK extensions for the GGN diagonal.

Includes
- ``DiagGGNExact``
- ``DiagGGNMC``
- ``BatchDiagGGNExact``
- ``BatchDiagGGNMC``

Shared settings are taken from `test.extensions.secondorder.secondorder_settings`.
Additional local cases can be defined here through ``LOCAL_SETTINGS``.
"""
from test.core.derivatives.utils import classification_targets, regression_targets
from test.extensions.automated_settings import make_simple_act_setting
from test.core.derivatives.utils import regression_targets
from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS
from test.utils.evaluation_mode import initialize_training_false_recursive

from torch import rand
from torch.nn import (
    RNN,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Flatten,
    MSELoss,
    Sequential,
)

from backpack.custom_module import branching
from backpack.custom_module.permute import Permute
from backpack.custom_module.reduce_tuple import ReduceTuple

SHARED_SETTINGS = SECONDORDER_SETTINGS
LOCAL_SETTINGS = [
    # RNN settings
    {
        "input_fn": lambda: rand(8, 5, 6),
        "module_fn": lambda: Sequential(
            Permute(1, 0, 2),
            RNN(input_size=6, hidden_size=3),
            ReduceTuple(index=0),
            Permute(1, 2, 0),
            Flatten(),
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((8, 3 * 5)),
    },
    {
        "input_fn": lambda: rand(2, 2, 9),
        "module_fn": lambda: Sequential(AdaptiveAvgPool1d((3,)), Flatten()),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((2, 2 * 3)),
    },
    {
        "input_fn": lambda: rand(2, 2, 6, 8),
        "module_fn": lambda: Sequential(AdaptiveAvgPool2d((3, 4)), Flatten()),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((2, 2 * 3 * 4)),
    },
    {
        "input_fn": lambda: rand(2, 2, 9, 5, 4),
        "module_fn": lambda: Sequential(AdaptiveAvgPool3d((3, 5, 2)), Flatten()),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((2, 2 * 3 * 5 * 2)),
    },
    {
        "input_fn": lambda: rand(2, 3, 4),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(BatchNorm1d(num_features=3), Flatten())
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((2, 4 * 3)),
    },
    {
        "input_fn": lambda: rand(3, 2, 4, 3),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(BatchNorm2d(num_features=2), Flatten())
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((3, 2 * 4 * 3)),
    },
    {
        "input_fn": lambda: rand(3, 3, 4, 1, 2),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(BatchNorm3d(num_features=3), Flatten())
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((3, 3 * 4 * 1 * 2)),
    },
]
###############################################################################
#                               Branched models                               #
###############################################################################
# TODO Integrate with LOCAL_SETTINGS after integrating branching with the extensions
# DiagGGNMC, BatchDiagGGNExact
LOCAL_SETTINGS += [
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
