"""Test configurations for `backpack.core.derivatives` for CONVOLUTIONal layers


Required entries:
    "module_fn" (callable): Contains a model constructed from `torch.nn` layers
    "input_fn" (callable): Used for specifying input function


Optional entries:
    "target_fn" (callable): Fetches the groundtruth/target classes 
                            of regression/classification task
    "loss_function_fn" (callable): Loss function used in the model
    "device" [list(torch.device)]: List of devices to run the test on.
    "id_prefix" (str): Prefix to be included in the test name.
    "seed" (int): seed for the random number for torch.rand
"""

import torch

CONVOLUTION_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "module_fn": lambda: torch.nn.Conv2d(
        in_channels=2,
        out_channels=3,
        kernel_size=2,
        bias=False,
        padding=1,
        stride=2,
        dilation=2,
    ),
    "input_fn": lambda: torch.rand(size=(3, 2, 7, 7)),
    "device": [torch.device("cpu")],  # optional
    "seed": 0,  # optional
    "id_prefix": "conv-example",  # optional
}
CONVOLUTION_SETTINGS.append(example)

###############################################################################
#                                test settings                                #
###############################################################################
CONVOLUTION_SETTINGS += [
    {
        "module_fn": lambda: torch.nn.Conv1d(
            in_channels=2, out_channels=3, kernel_size=2, padding=1, bias=False
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 7)),
    },
    {
        "module_fn": lambda: torch.nn.Conv1d(
            in_channels=3,
            out_channels=6,
            kernel_size=2,
            padding=2,
            padding_mode="zeros",
            stride=4,
            dilation=3,
        ),
        "input_fn": lambda: torch.rand(size=(1, 3, 8)),
    },
    {
        "module_fn": lambda: torch.nn.Conv1d(
            in_channels=2, out_channels=3, kernel_size=2, padding=1, groups=1
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 11)),
    },
    {
        "module_fn": lambda: torch.nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=2,
            padding=2,
            padding_mode="zeros",
            stride=4,
            dilation=3,
        ),
        "input_fn": lambda: torch.rand(size=(1, 3, 8, 8)),
    },
    {
        "module_fn": lambda: torch.nn.Conv2d(
            in_channels=2, out_channels=3, kernel_size=2, padding=1, groups=1
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 7, 7)),
    },
    {
        "module_fn": lambda: torch.nn.Conv3d(
            in_channels=2,
            out_channels=3,
            kernel_size=2,
            padding=2,
            bias=False,
            dilation=2,
            stride=2,
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 5, 7, 7)),
    },
    {
        "module_fn": lambda: torch.nn.Conv3d(
            in_channels=3,
            out_channels=6,
            kernel_size=2,
            padding=2,
            padding_mode="zeros",
            stride=4,
        ),
        "input_fn": lambda: torch.rand(size=(1, 3, 3, 4, 4)),
    },
    {
        "module_fn": lambda: torch.nn.Conv3d(
            in_channels=2, out_channels=3, kernel_size=2, padding=1, groups=1
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 3, 7, 7)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose1d(
            in_channels=2, out_channels=3, kernel_size=2, padding=1, bias=False
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 7)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose1d(
            in_channels=3,
            out_channels=6,
            kernel_size=2,
            padding=2,
            padding_mode="zeros",
            stride=4,
            dilation=3,
        ),
        "input_fn": lambda: torch.rand(size=(1, 3, 8)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose1d(
            in_channels=2, out_channels=3, kernel_size=2, padding=1, groups=1
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 11)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose2d(
            in_channels=2,
            out_channels=3,
            kernel_size=2,
            bias=False,
            padding=1,
            stride=2,
            dilation=2,
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 7, 7)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose2d(
            in_channels=3,
            out_channels=6,
            kernel_size=2,
            padding=2,
            padding_mode="zeros",
            stride=4,
            dilation=3,
        ),
        "input_fn": lambda: torch.rand(size=(1, 3, 8, 8)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose2d(
            in_channels=2, out_channels=3, kernel_size=2, padding=1, groups=1
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 7, 7)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose3d(
            in_channels=2,
            out_channels=3,
            kernel_size=2,
            padding=2,
            bias=False,
            dilation=2,
            stride=2,
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 5, 7, 7)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose3d(
            in_channels=3,
            out_channels=6,
            kernel_size=2,
            padding=2,
            padding_mode="zeros",
            stride=4,
        ),
        "input_fn": lambda: torch.rand(size=(1, 3, 3, 4, 4)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose3d(
            in_channels=2, out_channels=3, kernel_size=2, padding=1, groups=1
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 3, 7, 7)),
    },
]

# non-default hyperparameters
CONVOLUTION_SETTINGS += [
    {
        "module_fn": lambda: torch.nn.Conv1d(
            in_channels=2,
            out_channels=3,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=1,
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 7)),
        "id_prefix": "non-default-conv",
    },
    {
        "module_fn": lambda: torch.nn.Conv2d(
            in_channels=2,
            out_channels=3,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=1,
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 7, 7)),
        "id_prefix": "non-default-conv",
    },
    {
        "module_fn": lambda: torch.nn.Conv3d(
            in_channels=2,
            out_channels=3,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=1,
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 3, 7, 7)),
        "id_prefix": "non-default-conv",
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose1d(
            in_channels=2,
            out_channels=3,
            kernel_size=2,
            padding=0,
            stride=3,
            dilation=5,
            groups=1,
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 20)),
        "id_prefix": "non-default-conv",
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose2d(
            in_channels=2,
            out_channels=4,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=1,
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 9, 9)),
        "id_prefix": "non-default-conv",
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose3d(
            in_channels=2,
            out_channels=4,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=1,
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 9, 9, 9)),
        "id_prefix": "non-default-conv",
    },
]

CONVOLUTION_FAIL_SETTINGS = [
    # groups - 2
    {
        "module_fn": lambda: torch.nn.Conv1d(
            in_channels=4,
            out_channels=6,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=2,
        ),
        "input_fn": lambda: torch.rand(size=(3, 4, 7)),
        "id_prefix": "groups-2",
    },
    {
        "module_fn": lambda: torch.nn.Conv2d(
            in_channels=4,
            out_channels=6,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=2,
        ),
        "input_fn": lambda: torch.rand(size=(3, 4, 7, 7)),
        "id_prefix": "groups-2",
    },
    {
        "module_fn": lambda: torch.nn.Conv3d(
            in_channels=4,
            out_channels=6,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=2,
        ),
        "input_fn": lambda: torch.rand(size=(3, 4, 3, 7, 7)),
        "id_prefix": "groups-2",
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose1d(
            in_channels=4,
            out_channels=6,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=2,
        ),
        "input_fn": lambda: torch.rand(size=(3, 4, 7)),
        "id_prefix": "groups-2",
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose2d(
            in_channels=4,
            out_channels=6,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=2,
        ),
        "input_fn": lambda: torch.rand(size=(3, 4, 7, 7)),
        "id_prefix": "groups-2",
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose3d(
            in_channels=4,
            out_channels=6,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=2,
        ),
        "input_fn": lambda: torch.rand(size=(3, 4, 3, 7, 7)),
        "id_prefix": "groups-2",
    },
    # groups - 3
    {
        "module_fn": lambda: torch.nn.Conv1d(
            in_channels=6,
            out_channels=9,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=3,
        ),
        "input_fn": lambda: torch.rand(size=(3, 6, 7)),
        "id_prefix": "groups-3",
    },
    {
        "module_fn": lambda: torch.nn.Conv2d(
            in_channels=6,
            out_channels=9,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=3,
        ),
        "input_fn": lambda: torch.rand(size=(3, 6, 7, 7)),
        "id_prefix": "groups-3",
    },
    {
        "module_fn": lambda: torch.nn.Conv3d(
            in_channels=6,
            out_channels=9,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=3,
        ),
        "input_fn": lambda: torch.rand(size=(3, 6, 3, 7, 7)),
        "id_prefix": "groups-3",
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose1d(
            in_channels=6,
            out_channels=9,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=3,
        ),
        "input_fn": lambda: torch.rand(size=(3, 6, 7)),
        "id_prefix": "groups-3",
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose2d(
            in_channels=6,
            out_channels=9,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=3,
        ),
        "input_fn": lambda: torch.rand(size=(3, 6, 7, 7)),
        "id_prefix": "groups-3",
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose3d(
            in_channels=6,
            out_channels=9,
            kernel_size=2,
            padding=0,
            dilation=2,
            groups=3,
        ),
        "input_fn": lambda: torch.rand(size=(3, 6, 3, 7, 7)),
        "id_prefix": "groups-3",
    },
]
