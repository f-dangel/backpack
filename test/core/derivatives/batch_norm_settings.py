"""Test configurations for `backpack.core.derivatives` BatchNorm layers.

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

BATCH_NORM_SETTINGS = [
    {
        "module_fn": lambda: torch.nn.BatchNorm1d(num_features=7),
        "input_fn": lambda: torch.rand(size=(5, 7, 4)),
    },
    {
        "module_fn": lambda: torch.nn.BatchNorm1d(num_features=7),
        "input_fn": lambda: torch.rand(size=(5, 7)),
    },
    {
        "module_fn": lambda: torch.nn.BatchNorm2d(num_features=7),
        "input_fn": lambda: torch.rand(size=(5, 7, 3, 4)),
    },
    {
        "module_fn": lambda: torch.nn.BatchNorm3d(num_features=7),
        "input_fn": lambda: torch.rand(size=(5, 7, 3, 4, 2)),
    },
]
