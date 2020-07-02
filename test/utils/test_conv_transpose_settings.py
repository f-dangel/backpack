"""Test configurations for `backpack.test.utils` for CONVOLUTIONal layers
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

SETTINGS = []

###############################################################################
#                                   example                                   #
###############################################################################
example = {
    "module_fn": torch.nn.ConvTranspose2d,
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
SETTINGS.append(example)

###############################################################################
#                                test settings                                #
###############################################################################

SETTINGS += [
    {
        "module_fn": torch.nn.ConvTranspose1d,
        "module_kwargs": {
            "in_channels": 1,
            "out_channels": 2,
            "kernel_size": 2,
            "bias": False,
        },
        "input_kwargs": {"size": (1, 1, 3)},
    },
    {
        "module_fn": torch.nn.ConvTranspose1d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 1,
            "stride": 2,
            "bias": False,
        },
        "input_kwargs": {"size": (3, 2, 11)},
    },
    {
        "module_fn": torch.nn.ConvTranspose1d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 1,
            "stride": 2,
            "dilation": 2,
            "bias": False,
        },
        "input_kwargs": {"size": (3, 2, 11)},
    },
    {
        "module_fn": torch.nn.ConvTranspose1d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 4,
            "kernel_size": 2,
            "padding": 1,
            "stride": 2,
            "groups": 2,
            "bias": False,
        },
        "input_kwargs": {"size": (3, 2, 11)},
    },
    {
        "module_fn": torch.nn.ConvTranspose1d,
        "module_kwargs": {
            "in_channels": 3,
            "out_channels": 6,
            "kernel_size": 2,
            "padding": 1,
            "stride": 2,
            "groups": 3,
            "bias": False,
        },
        "input_kwargs": {"size": (3, 3, 11)},
    },
    {
        "module_fn": torch.nn.ConvTranspose2d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "bias": False,
            "padding": 1,
        },
        "input_kwargs": {"size": (3, 2, 7, 7)},
    },
    {
        "module_fn": torch.nn.ConvTranspose2d,
        "module_kwargs": {
            "in_channels": 3,
            "out_channels": 6,
            "kernel_size": 2,
            "stride": 4,
            "padding": 2,
            "bias": False,
            "padding_mode": "zeros",
            "dilation": 3,
        },
        "input_kwargs": {"size": (1, 3, 8, 8)},
    },
    {
        "module_fn": torch.nn.ConvTranspose2d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 6,
            "kernel_size": 2,
            "stride": 2,
            "padding": 1,
            "bias": False,
            "dilation": 2,
            "groups": 2,
        },
        "input_kwargs": {"size": (3, 2, 11, 13)},
    },
    {
        "module_fn": torch.nn.ConvTranspose2d,
        "module_kwargs": {
            "in_channels": 3,
            "out_channels": 6,
            "kernel_size": 2,
            "stride": 2,
            "padding": 1,
            "bias": False,
            "dilation": 2,
            "groups": 3,
        },
        "input_kwargs": {"size": (5, 3, 11, 13)},
    },
    {
        "module_fn": torch.nn.ConvTranspose2d,
        "module_kwargs": {
            "in_channels": 8,
            "out_channels": 15,
            "kernel_size": (3, 5),
            "stride": (2, 1),
            "padding": (4, 2),
            "bias": False,
            "dilation": (3, 1),
        },
        "input_kwargs": {"size": (10, 8, 25, 50)},
    },
    {
        "module_fn": torch.nn.ConvTranspose3d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "bias": False,
        },
        "input_kwargs": {"size": (3, 2, 2, 5, 5)},
    },
    {
        "module_fn": torch.nn.ConvTranspose3d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 1,
            "stride": 2,
            "bias": False,
        },
        "input_kwargs": {"size": (3, 2, 5, 13, 17)},
    },
    {
        "module_fn": torch.nn.ConvTranspose3d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": 2,
            "padding": 1,
            "stride": 2,
            "dilation": 2,
            "bias": False,
        },
        "input_kwargs": {"size": (3, 2, 5, 13, 17)},
    },
    {
        "module_fn": torch.nn.ConvTranspose3d,
        "module_kwargs": {
            "in_channels": 2,
            "out_channels": 4,
            "kernel_size": 2,
            "padding": 1,
            "stride": 2,
            "groups": 2,
            "bias": False,
        },
        "input_kwargs": {"size": (3, 2, 5, 13, 17)},
    },
    {
        "module_fn": torch.nn.ConvTranspose3d,
        "module_kwargs": {
            "in_channels": 3,
            "out_channels": 6,
            "kernel_size": 2,
            "padding": 1,
            "stride": 2,
            "groups": 3,
            "bias": False,
        },
        "input_kwargs": {"size": (3, 3, 5, 7, 7)},
    },
]


CONV_TRANSPOSE_2D_SETTINGS = [
    [torch.nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False), (1, 1, 3, 3)],
    [torch.nn.ConvTranspose2d(1, 2, kernel_size=2, bias=False), (1, 1, 3, 3)],
    [torch.nn.ConvTranspose2d(2, 1, kernel_size=2, bias=False), (1, 2, 3, 3)],
    [torch.nn.ConvTranspose2d(2, 2, kernel_size=2, bias=False), (1, 2, 3, 3)],
    [torch.nn.ConvTranspose2d(2, 3, kernel_size=2, bias=False), (3, 2, 11, 13)],
    [
        torch.nn.ConvTranspose2d(2, 3, kernel_size=2, padding=1, bias=False),
        (3, 2, 11, 13),
    ],
    [
        torch.nn.ConvTranspose2d(2, 3, kernel_size=2, padding=1, stride=2, bias=False),
        (3, 2, 11, 13),
    ],
    [
        torch.nn.ConvTranspose2d(
            2, 3, kernel_size=2, padding=1, stride=2, dilation=2, bias=False
        ),
        (3, 2, 11, 13),
    ],
]

CONV_TRANSPOSE_1D_SETTINGS = [
    [torch.nn.ConvTranspose1d(1, 1, kernel_size=2, bias=False), (1, 1, 3)],
    [torch.nn.ConvTranspose1d(1, 2, kernel_size=2, bias=False), (1, 1, 3)],
    [torch.nn.ConvTranspose1d(2, 1, kernel_size=2, bias=False), (1, 2, 3)],
    [torch.nn.ConvTranspose1d(2, 2, kernel_size=2, bias=False), (1, 2, 3)],
    [torch.nn.ConvTranspose1d(2, 3, kernel_size=2, bias=False), (3, 2, 11)],
    [torch.nn.ConvTranspose1d(2, 3, kernel_size=2, padding=1, bias=False), (3, 2, 11)],
    [
        torch.nn.ConvTranspose1d(2, 3, kernel_size=2, padding=1, stride=2, bias=False),
        (3, 2, 11),
    ],
    [
        torch.nn.ConvTranspose1d(
            2, 3, kernel_size=2, padding=1, stride=2, dilation=2, bias=False
        ),
        (3, 2, 11),
    ],
]

CONV_TRANSPOSE_3D_SETTINGS = [
    [torch.nn.ConvTranspose3d(1, 1, kernel_size=2, bias=False), (1, 1, 3, 3, 3)],
    [torch.nn.ConvTranspose3d(1, 2, kernel_size=2, bias=False), (1, 1, 3, 3, 3)],
    [torch.nn.ConvTranspose3d(2, 1, kernel_size=2, bias=False), (1, 2, 3, 3, 3)],
    [torch.nn.ConvTranspose3d(2, 2, kernel_size=2, bias=False), (1, 2, 3, 3, 3)],
    [torch.nn.ConvTranspose3d(2, 3, kernel_size=2, bias=False), (3, 2, 11, 13, 17)],
    [
        torch.nn.ConvTranspose3d(2, 3, kernel_size=2, padding=1, bias=False),
        (3, 2, 11, 13, 17),
    ],
    [
        torch.nn.ConvTranspose3d(2, 3, kernel_size=2, padding=1, stride=2, bias=False),
        (3, 2, 11, 13, 17),
    ],
    [
        torch.nn.ConvTranspose3d(
            2, 3, kernel_size=2, padding=1, stride=2, dilation=2, bias=False
        ),
        (3, 2, 11, 13, 17),
    ],
]
