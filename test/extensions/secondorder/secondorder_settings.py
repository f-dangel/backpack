"""Test configurations for `backpack.core.extensions.secondorder`
that is shared among the following secondorder methods:
- Diagonal of Gauss Newton
- Diagonal of Hessian
- MC Approximation of Diagonal of Gauss Newton


Required entries:
    "module_fn" (callable): Contains a model constructed from `torch.nn` layers
    "input_fn" (callable): Used for specifying input function
    "target_fn" (callable): Fetches the groundtruth/target classes 
                            of regression/classification task
    "loss_function_fn" (callable): Loss function used in the model

Optional entries:
    "device" [list(torch.device)]: List of devices to run the test on.
    "id_prefix" (str): Prefix to be included in the test name.
    "seed" (int): seed for the random number for torch.rand
"""


import torch
from test.core.derivatives.utils import classification_targets, regression_targets

from torch.nn import (
    ReLU,
    Sigmoid,
    Tanh,
    LogSigmoid,
    LeakyReLU,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)

from test.extensions.automated_settings import make_simple_act_setting, make_simple_cnn_setting


SECONDORDER_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "input_fn": lambda: torch.rand(3, 10),
    "module_fn": lambda: torch.nn.Sequential(torch.nn.Linear(10, 5)),
    "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(),
    "target_fn": lambda: classification_targets((3,), 5),
    "device": [torch.device("cpu")],
    "seed": 0,
    "id_prefix": "example",
}
SECONDORDER_SETTINGS.append(example)


SECONDORDER_SETTINGS += [
    # classification
    {
        "input_fn": lambda: torch.rand(3, 10),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Linear(10, 7), torch.nn.Linear(7, 5)
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 10),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Linear(10, 7), torch.nn.ReLU(), torch.nn.Linear(7, 5)
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    # Regression
    {
        "input_fn": lambda: torch.rand(3, 10),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Linear(10, 7), torch.nn.Sigmoid(), torch.nn.Linear(7, 5)
        ),
        "loss_function_fn": lambda: torch.nn.MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((3, 5)),
    },
]

###############################################################################
#                         test setting: Activation Layers                     #
###############################################################################
activations = [ReLU, Sigmoid, Tanh, LeakyReLU, LogSigmoid]

for act in activations:
    for bias in [True, False]:
        SECONDORDER_SETTINGS.append(make_simple_act_setting(act, bias=bias))

###############################################################################
#                         test setting: Convolutional Layers                  #
###############################################################################

SECONDORDER_SETTINGS += [
    # Conv1d
    make_simple_cnn_setting((3, 3, 7), Conv1d(3, 2, 2)),
    make_simple_cnn_setting((3, 3, 7), Conv1d(3, 2, 2, bias=False)),
    make_simple_cnn_setting(
        (3, 3, 8),
        Conv1d(3, 6, 2, stride=4, padding=2, padding_mode="zeros", dilation=3),
    ),
    make_simple_cnn_setting((3, 3, 7), Conv1d(3, 2, 2, padding=2, dilation=1, stride=2)),
    make_simple_cnn_setting((3, 2, 7), Conv1d(2, 3, 2, padding=0, dilation=2, groups=1)),
    # Conv2d
    make_simple_cnn_setting((3, 3, 7, 7), Conv2d(3, 2, 2)),
    make_simple_cnn_setting((3, 3, 7, 7), Conv2d(3, 2, 2, bias=False)),
    make_simple_cnn_setting(
        (3, 3, 8, 8),
        Conv2d(3, 6, 2, stride=4, padding=2, padding_mode="zeros", dilation=3),
    ),
    make_simple_cnn_setting((3, 3, 7, 7), Conv2d(3, 2, 2, padding=2, dilation=1, stride=2)),
    make_simple_cnn_setting((3, 2, 7, 7), Conv2d(2, 3, 2, padding=0, dilation=2, groups=1)),
    # Conv3d
    make_simple_cnn_setting((3, 3, 2, 7, 7), Conv3d(3, 2, 2)),
    make_simple_cnn_setting((3, 3, 2, 7, 7), Conv3d(3, 2, 2, bias=False)),
    make_simple_cnn_setting(
        (3, 3, 4, 8, 8),
        Conv3d(3, 6, 2, stride=4, padding=2, padding_mode="zeros", dilation=3),
    ),
    make_simple_cnn_setting((3, 3, 2, 7, 7), Conv3d(3, 2, 2, dilation=1, padding=2, stride=3)),
    make_simple_cnn_setting((3, 2, 3, 7, 7), Conv3d(2, 3, 2, dilation=2, padding=0)),
    # Conv1d
    make_simple_cnn_setting((3, 3, 7), ConvTranspose1d(3, 2, 2)),
    make_simple_cnn_setting((3, 3, 7), ConvTranspose1d(3, 2, 2, bias=False)),
    make_simple_cnn_setting(
        (3, 3, 8),
        ConvTranspose1d(3, 6, 2, stride=4, padding=2, padding_mode="zeros", dilation=3),
    ),
    make_simple_cnn_setting((3, 3, 7), ConvTranspose1d(3, 2, 2, padding=2, dilation=1, stride=2)),
    make_simple_cnn_setting((3, 2, 7), ConvTranspose1d(2, 3, 2, padding=0, dilation=2, groups=1)),
    # Conv2d
    make_simple_cnn_setting((3, 3, 7, 7), ConvTranspose2d(3, 2, 2)),
    make_simple_cnn_setting((3, 3, 7, 7), ConvTranspose2d(3, 2, 2, bias=False)),
    make_simple_cnn_setting(
        (3, 3, 8, 8),
        ConvTranspose2d(3, 6, 2, stride=4, padding=2, padding_mode="zeros", dilation=3),
    ),
    make_simple_cnn_setting(
        (3, 3, 7, 7), ConvTranspose2d(3, 2, 2, padding=2, dilation=1, stride=2)
    ),
    make_simple_cnn_setting(
        (3, 2, 7, 7), ConvTranspose2d(2, 3, 2, padding=0, dilation=2, groups=1)
    ),
    # Conv3d
    make_simple_cnn_setting((3, 3, 2, 7, 7), ConvTranspose3d(3, 2, 2)),
    make_simple_cnn_setting((3, 3, 2, 7, 7), ConvTranspose3d(3, 2, 2, bias=False)),
    make_simple_cnn_setting(
        (3, 3, 2, 5, 5), ConvTranspose3d(3, 2, 2, dilation=1, padding=2, stride=3)
    ),
    make_simple_cnn_setting((3, 2, 3, 7, 7), ConvTranspose3d(2, 3, 2, dilation=2, padding=0)),
]
