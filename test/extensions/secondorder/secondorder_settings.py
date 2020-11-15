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
from torch.nn import ReLU, Sigmoid, Tanh, LogSigmoid, LeakyReLU
from test.core.derivatives.utils import classification_targets, regression_targets
from test.extensions.automated_settings import make_simple_cnn_setting

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
        SECONDORDER_SETTINGS.append(make_simple_cnn_setting(act, bias=bias))

###############################################################################
#                         test setting: Convolutional Layers                  #
###############################################################################

SECONDORDER_SETTINGS += [
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
        "input_fn": lambda: torch.rand(3, 3, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, 2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(72, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 3, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(72, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 3, 8, 8),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv2d(
                3, 6, 2, stride=4, padding=2, padding_mode="zeros", dilation=3
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(54, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 3, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, 2, padding=0, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(18, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 2, 7, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 2, padding=0, dilation=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(75, 5),
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
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
