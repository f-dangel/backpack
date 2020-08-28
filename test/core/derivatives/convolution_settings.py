"""Test configurations for `backpack.core.derivatives` for CONVOLUTIONal layers


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

CONVOLUTION_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "module_fn": torch.nn.Conv2d,
    "module_kwargs": {
        "in_channels": 2,
        "out_channels": 3,
        "kernel_size": 2,
        "bias": False,
        "padding": 1,
        "stride": 2,
        "dilation": 2,
    },
    "input_kwargs": {"size": (3, 2, 7, 7)},
    "device": [torch.device("cpu")],  # optional
    "id_prefix": "layer-example",  # optional
    "seed": 0,  # optional
}
CONVOLUTION_SETTINGS.append(example)

###############################################################################
#                                test settings                                #
###############################################################################

CONVOLUTION_SETTINGS += [
    {
        "module_fn": torch.nn.Conv1d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "bias": False,
            "padding": 1,
        },
        "input_kwargs": {"size": (3, 2, 7)},
    },
    {
        "module_fn": torch.nn.Conv1d,
        "module_kwargs": {
            "in_channels": 3,
            "out_channels": 6,
            "kernel_size": 2,
            "stride": 4,
            "padding": 2,
            "padding_mode": "zeros",
            "dilation": 3,
        },
        "input_kwargs": {"size": (1, 3, 8)},
    },
    {
        "module_fn": torch.nn.Conv1d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 0,
            "dilation": 2,
            "groups": 1,
        },
        "input_kwargs": {"size": (3, 2, 7)},
        "id_prefix": "non-default-conv",
    },
    {
        "module_fn": torch.nn.Conv1d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 1,
            "groups": 1,
        },
        "input_kwargs": {"size": (3, 2, 11)},
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
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 1,
            "groups": 1,
        },
        "input_kwargs": {"size": (3, 2, 7, 7)},
    },
    {
        "module_fn": torch.nn.Conv2d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 0,
            "dilation": 2,
            "groups": 1,
        },
        "input_kwargs": {"size": (3, 2, 7, 7)},
        "id_prefix": "non-default-conv",
    },
    {
        "module_fn": torch.nn.Conv3d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "bias": False,
            "stride": 2,
            "padding": 1,
        },
        "input_kwargs": {"size": (3, 2, 3, 7, 7)},
    },
    {
        "module_fn": torch.nn.Conv3d,
        "module_kwargs": {
            "in_channels": 3,
            "out_channels": 6,
            "kernel_size": 2,
            "padding_mode": "zeros",
            "padding": 2,
        },
        "input_kwargs": {"size": (1, 3, 3, 4, 4)},
    },
    {
        "module_fn": torch.nn.Conv3d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 1,
            "groups": 1,
        },
        "input_kwargs": {"size": (3, 2, 3, 7, 7)},
    },
    {
        "module_fn": torch.nn.Conv3d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 0,
            "dilation": 2,
            "groups": 1,
        },
        "input_kwargs": {"size": (3, 2, 3, 7, 7)},
        "id_prefix": "non-default-conv",
    },
    {
        "module_fn": torch.nn.ConvTranspose1d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "bias": False,
            "padding": 1,
        },
        "input_kwargs": {"size": (3, 2, 7)},
    },
    {
        "module_fn": torch.nn.ConvTranspose1d,
        "module_kwargs": {
            "in_channels": 3,
            "out_channels": 6,
            "kernel_size": 2,
            "stride": 4,
            "padding": 2,
            "dilation": 3,
        },
        "input_kwargs": {"size": (1, 3, 7)},
    },
    {
        "module_fn": torch.nn.ConvTranspose1d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 1,
            "groups": 1,
        },
        "input_kwargs": {"size": (3, 2, 11)},
    },
    {
        "module_fn": torch.nn.ConvTranspose1d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 0,
            "stride": 3,
            "dilation": 5,
            "groups": 1,
        },
        "id_prefix": "non-default-conv",
        "input_kwargs": {"size": (3, 2, 20)},
    },
    {
        "module_fn": torch.nn.ConvTranspose2d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "bias": False,
            "kernel_size": 2,
            "padding": 1,
        },
        "input_kwargs": {"size": (3, 2, 7, 7)},
    },
    {
        "module_fn": torch.nn.ConvTranspose2d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 4,
            "kernel_size": 2,
            "padding": 0,
            "dilation": 2,
            "groups": 1,
        },
        "id_prefix": "non-default-conv",
        "input_kwargs": {"size": (3, 2, 9, 9)},
    },
    {
        "module_fn": torch.nn.ConvTranspose2d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 1,
            "stride": 2,
            "dilation": 2,
        },
        "input_kwargs": {"size": (3, 2, 7, 7)},
    },
    {
        "module_fn": torch.nn.ConvTranspose2d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 4,
            "kernel_size": 2,
            "padding": 1,
            "groups": 1,
        },
        "input_kwargs": {"size": (3, 2, 7, 7)},
    },
    {
        "module_fn": torch.nn.ConvTranspose3d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "bias": False,
            "stride": 2,
            "padding": 1,
        },
        "input_kwargs": {"size": (3, 2, 3, 7, 7)},
    },
    {
        "module_fn": torch.nn.ConvTranspose3d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 2,
            "dilation": 2,
            "stride": 2,
        },
        "input_kwargs": {"size": (3, 2, 5, 7, 7)},
    },
    {
        "module_fn": torch.nn.ConvTranspose3d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": (2, 2, 1),
            "groups": 1,
            "stride": (2, 2, 1),
        },
        "input_kwargs": {"size": (3, 2, 7, 7, 7)},
    },
    # # TODO: Fix groups ≠ 1
    # {
    #     "module_fn": torch.nn.ConvTranspose2d,
    #     "module_kwargs": {
    #         "in_channels": 6,
    #         "out_channels": 8,
    #         "kernel_size": 2,
    #         "padding": 1,
    #         "stride": 2,
    #         "dilation": 2,
    #         "groups": 2,
    #     },
    #     "input_kwargs": {"size": (3, 6, 11, 13)},
    # },
]
