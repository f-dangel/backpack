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

ALLOW_NEW_FORMAT = True
if ALLOW_NEW_FORMAT:
    new_format_example = {
        "module_fn": lambda: torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=2, bias=False, padding=1, stride=2, dilation=2),
        "input_fn": lambda: torch.rand(size=(3, 2, 7, 7)),
        "device": [torch.device("cpu")],  # optional
        "seed": 0,  # optional
        "id_prefix": "conv-new-format-example",  # optional
    }
    CONVOLUTION_SETTINGS.append(new_format_example)

###############################################################################
#                                test settings                                #
###############################################################################
if ALLOW_NEW_FORMAT:
    CONVOLUTION_SETTINGS += [
    {
        "module_fn": lambda: torch.nn.Conv1d(in_channels=2, out_channels=3, kernel_size=2, padding=1, bias=False),
        "input_fn": lambda: torch.rand(size=(3, 2, 7)),
    },
    {
        "module_fn": lambda: torch.nn.Conv1d(in_channels=3, out_channels=6, kernel_size=2, padding=2, padding_mode="zeros", stride=4, dilation=3),
        "input_fn": lambda: torch.rand(size=(1, 3, 8)),
    },
    {
        "module_fn": lambda: torch.nn.Conv1d(in_channels=2, out_channels=3, kernel_size=2, padding=1, groups=1),
        "input_fn": lambda: torch.rand(size=(3, 2, 11)),
    },
    {
        "module_fn": lambda: torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, padding=2, padding_mode="zeros", stride=4, dilation=3),
        "input_fn": lambda: torch.rand(size=(1, 3, 8, 8)),
    },
    {
        "module_fn": lambda: torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=2, padding=1, groups=1),
        "input_fn": lambda: torch.rand(size=(3, 2, 7, 7)),
    },
    {
        "module_fn": lambda: torch.nn.Conv3d(in_channels=2, out_channels=3, kernel_size=2, padding=2, bias=False, dilation=2, stride=2),
        "input_fn": lambda: torch.rand(size=(3, 2, 5, 7, 7)),
    },
    {
        "module_fn": lambda: torch.nn.Conv3d(in_channels=3, out_channels=6, kernel_size=2, padding=2, padding_mode="zeros", stride=4),
        "input_fn": lambda: torch.rand(size=(1, 3, 3, 4, 4)),
    },
    {
        "module_fn": lambda: torch.nn.Conv3d(in_channels=2, out_channels=3, kernel_size=2, padding=1, groups=1),
        "input_fn": lambda: torch.rand(size=(3, 2, 3, 7, 7)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose1d(in_channels=2, out_channels=3, kernel_size=2, padding=1, bias=False),
        "input_fn": lambda: torch.rand(size=(3, 2, 7)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose1d(in_channels=3, out_channels=6, kernel_size=2, padding=2, padding_mode="zeros", stride=4, dilation=3),
        "input_fn": lambda: torch.rand(size=(1, 3, 8)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose1d(in_channels=2, out_channels=3, kernel_size=2, padding=1, groups=1),
        "input_fn": lambda: torch.rand(size=(3, 2, 11)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose2d(in_channels=2, out_channels=3, kernel_size=2, bias=False, padding=1, stride=2, dilation=2),
        "input_fn": lambda: torch.rand(size=(3, 2, 7, 7)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose2d(in_channels=3, out_channels=6, kernel_size=2, padding=2, padding_mode="zeros", stride=4, dilation=3),
        "input_fn": lambda: torch.rand(size=(1, 3, 8, 8)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose2d(in_channels=2, out_channels=3, kernel_size=2, padding=1, groups=1),
        "input_fn": lambda: torch.rand(size=(3, 2, 7, 7)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose3d(in_channels=2, out_channels=3, kernel_size=2, padding=2, bias=False, dilation=2, stride=2),
        "input_fn": lambda: torch.rand(size=(3, 2, 5, 7, 7)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose3d(in_channels=3, out_channels=6, kernel_size=2, padding=2, padding_mode="zeros", stride=4),
        "input_fn": lambda: torch.rand(size=(1, 3, 3, 4, 4)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose3d(in_channels=2, out_channels=3, kernel_size=2, padding=1, groups=1),
        "input_fn": lambda: torch.rand(size=(3, 2, 3, 7, 7)),
    },       
    ]
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
