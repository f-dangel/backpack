"""Test configurations for `backpack.core.derivatives`.

Required entries:
    "module_fn" (torch.nn.Module): Module class of layer.
    "module_kwargs" (dict): Dictionary used for layer initialization.
        Leave empty if no arguments are required (e.g. for `torch.nn.ReLU`).
    "input_shape" (tuple): Dimensions of layer input.

Optional entries:
    "device" [list(torch.device)]: List of devices to run the test on.
    "id_prefix" (str): Prefix to be included in the test name.
"""

from test.core.derivatives.utils import classification_targets, regression_targets

import torch

SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "module_fn": torch.nn.Linear,
    "module_kwargs": {"in_features": 5, "out_features": 3, "bias": True},
    "input_fn": torch.rand,  # optional
    "input_kwargs": {"size": (10, 5)},  # needs to be modified for different "input_fn"
    "device": [torch.device("cpu")],  # optional
    "id_prefix": "config-example",  # optional
    "seed": 0,  # optional
}
SETTINGS.append(example)

loss_example = {
    "module_fn": torch.nn.MSELoss,
    "module_kwargs": {"reduction": "mean"},
    "input_fn": torch.rand,  # optional
    "input_kwargs": {"size": (8, 3)},  # needs to be modified for different "input_fn"
    "target_fn": regression_targets,
    "target_kwargs": {"size": (8, 3)},
    "device": [torch.device("cpu")],  # optional
    "id_prefix": "config-loss-example",  # optional
    "seed": 0,  # optional
}
SETTINGS.append(loss_example)

###############################################################################
#                                test settings                                #
###############################################################################

SETTINGS += [
    {
        "module_fn": torch.nn.Linear,
        "module_kwargs": {"in_features": 7, "out_features": 3, "bias": False,},
        "input_kwargs": {"size": (10, 7,)},
    },
    {
        "module_fn": torch.nn.ReLU,
        "module_kwargs": {},
        "input_kwargs": {"size": (10, 5)},
    },
    {
        "module_fn": torch.nn.Tanh,
        "module_kwargs": {},
        "input_kwargs": {"size": (1, 5, 6)},
    },
    {
        "module_fn": torch.nn.Sigmoid,
        "module_kwargs": {},
        "input_kwargs": {"size": (1, 5)},
    },
    {
        "module_fn": torch.nn.Conv2d,
        "module_kwargs": {
            "in_channels": 3,
            "out_channels": 3,
            "kernel_size": 4,
            "stride": 1,
        },
        "input_kwargs": {"size": (1, 3, 32, 32)},
    },
    {
        "module_fn": torch.nn.MaxPool2d,
        "module_kwargs": {"kernel_size": 2, "stride": 2},
        "input_kwargs": {"size": (1, 5, 32, 32)},
    },
    {
        "module_fn": torch.nn.AvgPool2d,
        "module_kwargs": {"kernel_size": 2},
        "input_kwargs": {"size": (1, 3, 32, 32)},
    },
    # loss functions
    {
        "module_fn": torch.nn.CrossEntropyLoss,
        "module_kwargs": {"reduction": "mean"},
        "input_kwargs": {"size": (2, 4)},
        "target_fn": classification_targets,
        "target_kwargs": {"size": (2,), "num_classes": 3},
    },
    {
        "module_fn": torch.nn.MSELoss,
        "module_kwargs": {"reduction": "mean"},
        "input_kwargs": {"size": (5, 1)},
        "target_fn": regression_targets,
        "target_kwargs": {"size": (5, 1)},
    },
    ## CrossEntropyLoss
    ### reduction 'mean'
    {
        "module_fn": torch.nn.CrossEntropyLoss,
        "module_kwargs": {"reduction": "mean"},
        "input_kwargs": {"size": (2, 3)},
        "target_fn": classification_targets,
        "target_kwargs": {"size": (2,), "num_classes": 3},
    },
    {
        "module_fn": torch.nn.CrossEntropyLoss,
        "module_kwargs": {"reduction": "mean"},
        "input_kwargs": {"size": (8, 2)},
        "target_fn": classification_targets,
        "target_kwargs": {"size": (8,), "num_classes": 2},
    },
    ### reduction 'sum'
    {
        "module_fn": torch.nn.CrossEntropyLoss,
        "module_kwargs": {"reduction": "sum"},
        "input_kwargs": {"size": (2, 3)},
        "target_fn": classification_targets,
        "target_kwargs": {"size": (2,), "num_classes": 3},
    },
    {
        "module_fn": torch.nn.CrossEntropyLoss,
        "module_kwargs": {"reduction": "sum"},
        "input_kwargs": {"size": (8, 2)},
        "target_fn": classification_targets,
        "target_kwargs": {"size": (8,), "num_classes": 2},
    },
    ### reduction 'none': while non-scalar outputs are not
    ### supported, a single number as output should be fine
    {
        "module_fn": torch.nn.CrossEntropyLoss,
        "module_kwargs": {"reduction": "none"},
        "input_kwargs": {"size": (1, 1)},
        "target_fn": classification_targets,
        "target_kwargs": {"size": (1,), "num_classes": 3},
    },
    ## MSELoss
    ### reduction 'mean'
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
    ### reduction 'sum'
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
    ### reduction 'none': while non-scalar outputs are not
    ### supported, a single number as output should be fine
    {
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
