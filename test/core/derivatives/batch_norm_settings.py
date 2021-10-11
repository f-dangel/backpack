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
from test.utils.evaluation_mode import initialize_batch_norm_eval

from torch import rand
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

BATCH_NORM_SETTINGS = [
    {
        "module_fn": lambda: BatchNorm1d(num_features=7),
        "input_fn": lambda: rand(size=(5, 7)),
    },
    {
        "module_fn": lambda: BatchNorm1d(num_features=7),
        "input_fn": lambda: rand(size=(5, 7, 4)),
    },
    {
        "module_fn": lambda: BatchNorm2d(num_features=7),
        "input_fn": lambda: rand(size=(5, 7, 3, 4)),
    },
    {
        "module_fn": lambda: BatchNorm3d(num_features=3),
        "input_fn": lambda: rand(size=(5, 3, 3, 4, 2)),
    },
    {
        "module_fn": lambda: initialize_batch_norm_eval(BatchNorm1d(num_features=7)),
        "input_fn": lambda: rand(size=(5, 7)),
        "id_prefix": "training=False",
    },
    {
        "module_fn": lambda: initialize_batch_norm_eval(BatchNorm1d(num_features=7)),
        "input_fn": lambda: rand(size=(5, 7, 4)),
        "id_prefix": "training=False",
    },
    {
        "module_fn": lambda: initialize_batch_norm_eval(BatchNorm2d(num_features=7)),
        "input_fn": lambda: rand(size=(5, 7, 3, 4)),
        "id_prefix": "training=False",
    },
    {
        "module_fn": lambda: initialize_batch_norm_eval(BatchNorm3d(num_features=7)),
        "input_fn": lambda: rand(size=(5, 7, 3, 4, 2)),
        "id_prefix": "training=False",
    },
]
