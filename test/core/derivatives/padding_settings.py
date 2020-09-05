"""Test configurations for `backpack.core.derivatives` for PADDING Layers

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

PADDING_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "module_fn": lambda: torch.nn.ZeroPad2d(2),
    "input_fn": lambda: torch.rand(size=(4, 3, 4, 5)),
    "device": [torch.device("cpu")],  # optional
    "seed": 0,  # optional
    "id_prefix": "padding-example",  # optional
}
PADDING_SETTINGS.append(example)
###############################################################################
#                                test settings                                #
###############################################################################

PADDING_SETTINGS += [
    {
        "module_fn": lambda: torch.nn.ZeroPad2d(3),
        "input_fn": lambda: torch.rand(size=(2, 3, 4, 5)),
    },
    {
        "module_fn": lambda: torch.nn.ZeroPad2d(5),
        "input_fn": lambda: torch.rand(size=(4, 3, 4, 5)),
    },
]
