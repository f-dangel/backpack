"""Test configurations for `backpack.core.derivatives` for Loss functions

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

from test.core.derivatives.utils import classification_targets, regression_targets

import torch

LOSS_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################


example = {
    "module_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
    "input_fn": lambda: torch.rand(size=(2, 4)),
    "target_fn": lambda: classification_targets(size=(2,), num_classes=2),
    "device": [torch.device("cpu")],  # optional
    "seed": 0,  # optional
    "id_prefix": "loss-example",  # optional
}
LOSS_SETTINGS.append(example)


LOSS_SETTINGS += [
    {
        "module_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "input_fn": lambda: torch.rand(size=(2, 4)),
        "target_fn": lambda: classification_targets(size=(2,), num_classes=2),
    },
    {
        "module_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "input_fn": lambda: torch.rand(size=(8, 4)),
        "target_fn": lambda: classification_targets(size=(8,), num_classes=2),
    },
    {
        "module_fn": lambda: torch.nn.CrossEntropyLoss(reduction="none"),
        "input_fn": lambda: torch.rand(size=(1, 1)),
        "target_fn": lambda: classification_targets(size=(1,), num_classes=1),
    },
    {
        "module_fn": lambda: torch.nn.MSELoss(reduction="mean"),
        "input_fn": lambda: torch.rand(size=(5, 1)),
        "target_fn": lambda: regression_targets(size=(5, 1)),
    },
    {
        "module_fn": lambda: torch.nn.MSELoss(reduction="sum"),
        "input_fn": lambda: torch.rand(size=(5, 3)),
        "target_fn": lambda: regression_targets(size=(5, 3)),
    },
    {
        "module_fn": lambda: torch.nn.MSELoss(reduction="none"),
        "input_fn": lambda: torch.rand(size=(1, 1)),
        "target_fn": lambda: regression_targets(size=(1, 1)),
    },
]


LOSS_FAIL_SETTINGS = [
    # non-scalar outputs are not supported
    {
        "module_fn": lambda: torch.nn.CrossEntropyLoss(reduction="none"),
        "input_fn": lambda: torch.rand(size=(5, 1)),
        "target_fn": lambda: classification_targets(size=(8,), num_classes=1),
    },
    {
        "module_fn": lambda: torch.nn.MSELoss(reduction="none"),
        "input_fn": lambda: torch.rand(size=(5, 1)),
        "target_fn": lambda: regression_targets(size=(5, 1)),
    },
]
