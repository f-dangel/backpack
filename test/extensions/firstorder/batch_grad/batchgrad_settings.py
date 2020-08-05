"""Test configurations for `backpack.core.extensions.firstorder` for batch_grad


Required entries:
    "module_fn" (callable): Contains a model constructed from torch.nn layers
    "input_fn" (callable): Used for specifying input function
    "target_fn" (callable): Fetches the groundtruth/target classes of regression task
    "loss_function_fn" (callable): Loss function used in the model

Optional entries:
    "device" [list(torch.device)]: List of devices to run the test on.
    "id_prefix" (str): Prefix to be included in the test name.
    "seed" (int): seed for the random number for torch.rand
"""


import torch
from test.core.derivatives.utils import classification_targets

BATCHGRAD_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "input_fn": lambda: torch.rand(3, 10),
    "module_fn": lambda: torch.nn.Sequential(
        torch.nn.Linear(10, 7), torch.nn.Sigmoid(), torch.nn.Linear(7, 5)
    ),
    "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
    "target_fn": lambda: classification_targets((3,), 5),
    "device": [torch.device("cpu")],
    "seed": 0,
    "id_prefix": "prototype",
}
BATCHGRAD_SETTINGS.append(example)

###############################################################################
#                                   test setting                              #
###############################################################################

BATCHGRAD_SETTINGS += [{
    "input_fn": lambda: torch.rand(3, 10),
    "module_fn": lambda: torch.nn.Sequential(
        torch.nn.Linear(10, 7), torch.nn.Sigmoid(), torch.nn.Linear(7, 5)
    ),
    "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
    "target_fn": lambda: classification_targets((3,), 5),
    "device": [torch.device("cpu")],
    "seed": 0,
    "id_prefix": "prototype",
}]