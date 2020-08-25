"""Test configurations for `backpack.core.extensions.firstorder`
that is shared among the following firstorder methods:
- batch_grad
- batch_l2_grad
- sum_grad_sqaured
- variance


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

FIRSTORDER_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "input_fn": lambda: torch.rand(3, 10),
    "module_fn": lambda: torch.nn.Sequential(torch.nn.Linear(10, 5)),
    "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
    "target_fn": lambda: classification_targets((3,), 5),
    "device": [torch.device("cpu")],
    "seed": 0,
    "id_prefix": "example",
}
FIRSTORDER_SETTINGS.append(example)

###############################################################################
#                         test setting: Linear Layers                         #
###############################################################################

FIRSTORDER_SETTINGS += [
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
#                         test setting: Convolutional Layers                  #
###############################################################################

FIRSTORDER_SETTINGS += [
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
        "input_fn": lambda: torch.rand(3, 1, 7),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.ConvTranspose1d(1, 2, 2),
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
            torch.nn.ConvTranspose1d(3, 2, 2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 5),
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
]
