"""Test configurations for `backpack.core.derivatives` for Pooling Layers

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

pooling_SETTINGS = []

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
pooling_SETTINGS.append(example)


###############################################################################
#                                test settings                                #
###############################################################################
pooling_SETTINGS += [
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
    {
        "module_fn": torch.nn.ZeroPad2d,
        "module_kwargs": {"padding": 2},
        "input_kwargs": {"size": (4, 3, 4, 5)},
    },
    {
        "module_fn": torch.nn.ZeroPad2d,
        "module_kwargs": {"padding": 5},
        "input_kwargs": {"size": (4, 3, 4, 4)},
    },
]
