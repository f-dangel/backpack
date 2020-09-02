"""Test configurations for `backpack.core.derivatives` for Loss functions

Required entries:
    "module_fn" (torch.nn.Module): Module class of Loss functions
    "module_kwargs" (dict): Dictionary used for loss function parameters
    "target_fn" (function): Fetches the groundtruth/target classes of regression task
    "target_kwargs" (dict): Dictonary used for target function parameters. Eg: Size

Optional entries:
    "device" [list(torch.device)]: List of devices to run the test on.
    "id_prefix" (str): Prefix to be included in the test name.
"""

from test.core.derivatives.utils import classification_targets, regression_targets

import torch

LOSS_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "module_fn": torch.nn.MSELoss,
    "module_kwargs": {"reduction": "mean"},
    "input_fn": torch.rand,  # optional
    "input_kwargs": {"size": (8, 3)},  # needs to be modified for different "input_fn"
    "target_fn": regression_targets,
    "target_kwargs": {"size": (8, 3)},
    "device": [torch.device("cpu")],  # optional
    "id_prefix": "loss",  # optional
    "seed": 0,  # optional
}
LOSS_SETTINGS.append(example)

ALLOW_NEW_FORMAT = True
if ALLOW_NEW_FORMAT:
    new_format_example = {
        "module_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "input_fn": lambda: torch.rand(size=(2, 4)),
        "target_fn": lambda: classification_targets(size=(2,), num_classes=2),
        "device": [torch.device("cpu")],  # optional
        "seed": 0,  # optional
        "id_prefix": "loss-new-format-example",  # optional
    }
    LOSS_SETTINGS.append(new_format_example)

if ALLOW_NEW_FORMAT:
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
        "target_fn": lambda: regression_targets(size=(5,1)),   
        },
        {
        "module_fn": lambda: torch.nn.MSELoss(reduction="sum"),
        "input_fn": lambda: torch.rand(size=(5, 3)),
        "target_fn": lambda: regression_targets(size=(5,3)),   
        },
        {
        "module_fn": lambda: torch.nn.MSELoss(reduction="none"),
        "input_fn": lambda: torch.rand(size=(1, 1)),
        "target_fn": lambda: regression_targets(size=(1,1)),   
        },
        ]

LOSS_SETTINGS += [
    {
        "module_fn": torch.nn.CrossEntropyLoss,
        "module_kwargs": {"reduction": "mean"},
        "input_kwargs": {"size": (2, 4)},
        "target_fn": classification_targets,
        "target_kwargs": {"size": (2,), "num_classes": 2},
    },
    {
        "module_fn": torch.nn.MSELoss,
        "module_kwargs": {"reduction": "mean"},
        "input_kwargs": {"size": (5, 1)},
        "target_fn": regression_targets,
        "target_kwargs": {"size": (5, 1)},
    },
    {
        "module_fn": torch.nn.CrossEntropyLoss,
        "module_kwargs": {"reduction": "mean"},
        "input_kwargs": {"size": (2, 3)},
        "target_fn": classification_targets,
        "target_kwargs": {"size": (2,), "num_classes": 2},
    },
    {
        "module_fn": torch.nn.CrossEntropyLoss,
        "module_kwargs": {"reduction": "mean"},
        "input_kwargs": {"size": (8, 2)},
        "target_fn": classification_targets,
        "target_kwargs": {"size": (8,), "num_classes": 2},
    },
    {
        "module_fn": torch.nn.CrossEntropyLoss,
        "module_kwargs": {"reduction": "sum"},
        "input_kwargs": {"size": (2, 3)},
        "target_fn": classification_targets,
        "target_kwargs": {"size": (2,), "num_classes": 2},
    },
    {
        "module_fn": torch.nn.CrossEntropyLoss,
        "module_kwargs": {"reduction": "sum"},
        "input_kwargs": {"size": (8, 2)},
        "target_fn": classification_targets,
        "target_kwargs": {"size": (8,), "num_classes": 2},
    },
    {
        # reduction 'none': while non-scalar outputs are not supported,
        # a single number as output should be fine
        "module_fn": torch.nn.CrossEntropyLoss,
        "module_kwargs": {"reduction": "none"},
        "input_kwargs": {"size": (1, 1)},
        "target_fn": classification_targets,
        "target_kwargs": {"size": (1,), "num_classes": 1},
    },
    {
        "module_fn": torch.nn.MSELoss,
        "module_kwargs": {"reduction": "mean"},
        "input_kwargs": {"size": (5, 1)},
        "target_fn": regression_targets,
        "target_kwargs": {"size": (5, 1)},
    },
    {
        "module_fn": torch.nn.MSELoss,
        "module_kwargs": {"reduction": "mean"},
        "input_kwargs": {"size": (5, 2)},
        "target_fn": regression_targets,
        "target_kwargs": {"size": (5, 2)},
    },
    {
        "module_fn": torch.nn.MSELoss,
        "module_kwargs": {"reduction": "sum"},
        "input_kwargs": {"size": (5, 1)},
        "target_fn": regression_targets,
        "target_kwargs": {"size": (5, 1)},
    },
    {
        "module_fn": torch.nn.MSELoss,
        "module_kwargs": {"reduction": "sum"},
        "input_kwargs": {"size": (5, 2)},
        "target_fn": regression_targets,
        "target_kwargs": {"size": (5, 2)},
    },
    {
        # reduction 'none': while non-scalar outputs are not supported,
        # a single number as output should be fine
        "module_fn": torch.nn.MSELoss,
        "module_kwargs": {"reduction": "none"},
        "input_kwargs": {"size": (1, 1)},
        "target_fn": regression_targets,
        "target_kwargs": {"size": (1, 1)},
    },
]

LOSS_FAIL_SETTINGS = [
    # non-scalar outputs are not supported
    {
        "module_fn": torch.nn.CrossEntropyLoss,
        "module_kwargs": {"reduction": "none"},
        "input_kwargs": {"size": (8, 2)},
        "target_fn": classification_targets,
        "target_kwargs": {"size": (8,), "num_classes": 2},
    },
    {
        "module_fn": torch.nn.MSELoss,
        "module_kwargs": {"reduction": "none"},
        "input_kwargs": {"size": (5, 1)},
        "target_fn": regression_targets,
        "target_kwargs": {"size": (5, 1)},
    },
]
