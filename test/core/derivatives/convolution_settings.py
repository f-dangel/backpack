"""Test configurations for `backpack.core.derivatives` for convolutional layers


Required entries:
    "module_fn" (torch.nn.Module): Module class of layer.
    "module_kwargs" (dict): Dictionary used for layer initialization.
    "input_kwargs" (dict): Parameters required by torch.nn.Module class
        "size": Requires a 4-Dimensional tensor

Optional entries:
    "device" [list(torch.device)]: List of devices to run the test on.
    "id_prefix" (str): Prefix to be included in the test name.
"""

import torch

convolution_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "module_fn": torch.nn.Linear,
    "module_kwargs": {"in_features": 5, "out_features": 3, "bias": True},
    "input_fn": torch.rand,  # optional
    "input_kwargs": {"size": (10, 5)},  # needs to be modified for different "input_fn"
    "device": [torch.device("cpu")],  # optional
    "id_prefix": "layer-example",  # optional
    "seed": 0,  # optional
}
convolution_SETTINGS.append(example)

###############################################################################
#                                test settings                                #
###############################################################################

convolution_SETTINGS += [
    {
        "module_fn": torch.nn.Conv2d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 1,
        },
        "input_kwargs": {"size": (3, 2, 11, 13)},
    },
    {
        "module_fn": torch.nn.Conv2d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 1,
            "stride": 2,
            "dilation": 2,
        },
        "input_kwargs": {"size": (3, 2, 11, 13)},
    },
    {
        "module_fn": torch.nn.Conv2d,
        "module_kwargs": {
            "in_channels": 3,
            "out_channels": 6,
            "kernel_size": 2,
            "stride": 4,
            "padding": 2,
            "padding_mode": "zeros",
            "dilation": 3,
        },
        "input_kwargs": {"size": (1, 3, 8, 8)},
    },
    {
        "module_fn": torch.nn.Conv2d,
        "module_kwargs": {
            "in_channels": 4,
            "out_channels": 4,
            "kernel_size": 2,
            "stride": 2,
            "padding": 2,
            "padding_mode": "reflect",
        },
        "input_kwargs": {"size": (1, 4, 4, 4)},
    },
    {
        "module_fn": torch.nn.Conv2d,
        "module_kwargs": {
            "in_channels": 3,
            "out_channels": 2,
            "kernel_size": 2,
            "stride": 1,
            "padding": 2,
            "padding_mode": "replicate",
            "bias": True,
        },
        "input_kwargs": {"size": (1, 3, 4, 4)},
    },
    {
        "module_fn": torch.nn.Conv2d,
        "module_kwargs": {
            "in_channels": 3,
            "out_channels": 6,
            "kernel_size": 2,
            "stride": 2,
            "padding": 2,
            "padding_mode": "cirular",
        },
        "input_kwargs": {"size": (1, 3, 8, 8)},
    },
    {
        "module_fn": torch.nn.Conv2d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 1,
            "groups": 1,
        },
        "input_kwargs": {"size": (3, 2, 11, 13)},
    },
]
