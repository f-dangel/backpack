"""Test configurations for `backpack.core.derivatives` for adaptive pooling layers.

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

POOLING_ADAPTIVE_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "module_fn": lambda: torch.nn.AdaptiveAvgPool2d(output_size=(2, 2)),
    "input_fn": lambda: torch.rand(size=(1, 3, 8, 8)),
    "device": [torch.device("cpu")],  # optional
    "seed": 0,  # optional
    "id_prefix": "pooling-example",  # optional
}
POOLING_ADAPTIVE_SETTINGS.append(example)
