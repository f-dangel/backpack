"""Test configurations for `backpack.core.derivatives` for PADDING Layers

Required entries:
    "module_fn" (torch.nn.Module): Module class of layer.
    "module_kwargs" (dict): Dictionary used for layer initialization.
        Leave empty if no arguments are required (e.g. for `torch.nn.ReLU`).
    "input_kwargs" (tuple): Dimensions of layer input.

Optional entries:
    "device" [list(torch.device)]: List of devices to run the test on.
    "id_prefix" (str): Prefix to be included in the test name.
"""

import torch

PADDING_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "module_fn": torch.nn.ZeroPad2d,
    "module_kwargs": {"padding": 2},
    "input_kwargs": {"size": (4, 3, 4, 5)},
    "device": [torch.device("cpu")],  # optional
    "id_prefix": "pad",  # optional
    "seed": 0,  # optional
}
PADDING_SETTINGS.append(example)


###############################################################################
#                                test settings                                #
###############################################################################

PADDING_SETTINGS += [
    {
        "module_fn": torch.nn.ZeroPad2d,
        "module_kwargs": {"padding": 3},
        "input_kwargs": {"size": (4, 3, 4, 5)},
    },
    {
        "module_fn": torch.nn.ZeroPad2d,
        "module_kwargs": {"padding": 5},
        "input_kwargs": {"size": (4, 3, 4, 4)},
    },
]
