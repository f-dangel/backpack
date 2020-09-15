"""Test configurations for `backpack.test.utils` for Transpose CONVOLUTIONal layers
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

SETTINGS = []

###############################################################################
#                                   example                                   #
###############################################################################
example = {
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
    "device": [torch.device("cpu")],  # optional
    "seed": 0,  # optional
    "id_prefix": "conv-example",  # optional
}
SETTINGS.append(example)

###############################################################################
#                                test settings                                #
###############################################################################

SETTINGS += [
    {
        "module_fn": lambda: torch.nn.ConvTranspose1d(
            in_channels=2, out_channels=3, kernel_size=2, padding=1, bias=False
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 7)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose1d(
            in_channels=2,
            out_channels=3,
            kernel_size=2,
            padding=1,
            stride=2,
            dilation=2,
            bias=False,
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 11)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose1d(
            in_channels=2,
            out_channels=4,
            kernel_size=2,
            padding=1,
            groups=2,
            stride=2,
            bias=False,
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 11)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose1d(
            in_channels=3,
            out_channels=6,
            kernel_size=2,
            padding=1,
            stride=2,
            groups=3,
            bias=False,
        ),
        "input_fn": lambda: torch.rand(size=(3, 3, 11)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose2d(
            in_channels=3,
            out_channels=6,
            kernel_size=2,
            padding=2,
            bias=False,
            padding_mode="zeros",
            stride=4,
            dilation=3,
        ),
        "input_fn": lambda: torch.rand(size=(1, 3, 8, 8)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose2d(
            in_channels=2,
            out_channels=6,
            kernel_size=2,
            padding=1,
            groups=2,
            bias=False,
            dilation=2,
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 11, 13)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose2d(
            in_channels=8,
            out_channels=15,
            kernel_size=(3, 5),
            stride=(2, 1),
            padding=(4, 2),
            bias=False,
            dilation=(3, 1),
        ),
        "input_fn": lambda: torch.rand(size=(10, 8, 25, 50)),
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
            bias=False,
        ),
        "input_fn": lambda: torch.rand(size=(1, 3, 3, 4, 4)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose3d(
            in_channels=2,
            out_channels=3,
            kernel_size=2,
            padding=1,
            stride=2,
            dilation=2,
            bias=False,
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 5, 13, 17)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose3d(
            in_channels=2,
            out_channels=4,
            kernel_size=2,
            padding=1,
            stride=2,
            groups=2,
            bias=False,
        ),
        "input_fn": lambda: torch.rand(size=(3, 2, 5, 13, 17)),
    },
    {
        "module_fn": lambda: torch.nn.ConvTranspose3d(
            in_channels=3,
            out_channels=6,
            kernel_size=2,
            padding=1,
            stride=2,
            groups=3,
            bias=False,
        ),
        "input_fn": lambda: torch.rand(size=(3, 3, 5, 7, 7)),
    },
]
