"""Test configurations for `backpack.core.derivatives` for POOLING Layers

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

POOLING_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
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
    "id_prefix": "pooling-example",  # optional
}
POOLING_SETTINGS.append(example)
###############################################################################
#                                test settings                                #
###############################################################################

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
