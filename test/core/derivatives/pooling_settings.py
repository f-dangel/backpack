"""Test configurations for `backpack.core.derivatives` for POOLING Layers

Required entries:
    "module_fn" (torch.nn.Module): Module class of layer.
    "module_kwargs" (dict): Dictionary used for layer initialization.
        Leave empty if no arguments are required (e.g. for `torch.nn.ReLU`).
    "input_shape" (tuple): Dimensions of layer input.
        Must be a 4-Dimensional tensor

Optional entries:
    "device" [list(torch.device)]: List of devices to run the test on.
    "id_prefix" (str): Prefix to be included in the test name.
"""

import torch

POOLING_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "module_fn": torch.nn.MaxPool2d,
    "module_kwargs": {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
    },
    "input_fn": torch.rand,  # optional
    "input_kwargs": {"size": (1, 3, 8, 8)},
    "device": [torch.device("cpu")],  # optional
    "id_prefix": "pool",  # optional
    "seed": 0,  # optional
}
POOLING_SETTINGS.append(example)

ALLOW_NEW_FORMAT = True
if ALLOW_NEW_FORMAT:
    new_format_example = {
        "module_fn": lambda: torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        ),
        "input_fn": lambda: torch.rand(size=(1, 3, 8, 8)),
        "device": [torch.device("cpu")],  # optional
        "seed": 0,  # optional
        "id_prefix": "pooling-new-format-example",  # optional
    }
    POOLING_SETTINGS.append(new_format_example)
###############################################################################
#                                test settings                                #
###############################################################################
ALLOW_NEW_FORMAT = True
if ALLOW_NEW_FORMAT:
    POOLING_SETTINGS += [
        {
            "module_fn": lambda: torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            "input_fn": lambda: torch.rand(size=(1, 3, 8, 8)),
        },
        {
            "module_fn": lambda: torch.nn.MaxPool2d(kernel_size=2, ceil_mode=True),
            "input_fn": lambda: torch.rand(size=(1, 2, 4, 4)),
        },
        {
            "module_fn": lambda: torch.nn.MaxPool2d(
                kernel_size=3, stride=(2, 1), dilation=3
            ),
            "input_fn": lambda: torch.rand(size=(3, 2, 9, 9)),
        },
        {
            "module_fn": lambda: torch.nn.AvgPool2d(
                kernel_size=3, stride=(2, 1), padding=1
            ),
            "input_fn": lambda: torch.rand(size=(3, 2, 9, 9)),
        },
        {
            "module_fn": lambda: torch.nn.AvgPool2d(kernel_size=4, padding=2),
            "input_fn": lambda: torch.rand(size=(3, 2, 9, 9)),
        },
    ]
POOLING_SETTINGS += [
    {
        "module_fn": torch.nn.AvgPool2d,
        "module_kwargs": {"kernel_size": 2},
        "input_kwargs": {"size": (1, 3, 8, 8)},
    },
    {
        "module_fn": torch.nn.AvgPool2d,
        "module_kwargs": {"kernel_size": 4, "padding": 2},
        "input_kwargs": {"size": (3, 2, 15, 13)},
    },
    {
        "module_fn": torch.nn.AvgPool2d,
        "module_kwargs": {"kernel_size": 3, "stride": (2, 1), "padding": 1},
        "input_kwargs": {"size": (3, 2, 15, 13)},
    },
    {
        "module_fn": torch.nn.MaxPool2d,
        "module_kwargs": {"kernel_size": 2, "stride": 2, "padding": 1},
        "input_kwargs": {"size": (1, 5, 8, 8)},
    },
    {
        "module_fn": torch.nn.MaxPool2d,
        "module_kwargs": {"kernel_size": 2, "ceil_mode": True},
        "input_kwargs": {"size": (1, 2, 4, 4)},
    },
    {
        "module_fn": torch.nn.MaxPool2d,
        "module_kwargs": {"kernel_size": 3, "stride": (2, 1), "dilation": 3},
        "input_kwargs": {"size": (3, 2, 9, 9)},
    },
]
