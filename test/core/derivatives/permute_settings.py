"""Test configurations for `backpack.core.derivatives` Permute.

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

from backpack.custom_module.permute import Permute

PERMUTE_SETTINGS = [
    {
        "module_fn": lambda: Permute(0, 1, 2),
        "input_fn": lambda: torch.rand(size=(1, 2, 3)),
    },
    {
        "module_fn": lambda: Permute(0, 2, 1),
        "input_fn": lambda: torch.rand(size=(4, 3, 2)),
    },
    {
        "module_fn": lambda: Permute(0, 3, 1, 2),
        "input_fn": lambda: torch.rand(size=(5, 4, 3, 2)),
    },
]
