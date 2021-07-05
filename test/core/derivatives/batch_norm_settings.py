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
from typing import Union

from torch import rand, rand_like
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d


def _initialize_training_false(
    module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]
) -> Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]:
    module.running_mean = rand_like(module.running_mean)
    module.running_var = rand_like(module.running_var)
    module.weight.data = rand_like(module.weight)
    module.bias.data = rand_like(module.bias)
    return module.train(False)


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
        "module_fn": lambda: BatchNorm3d(num_features=7),
        "input_fn": lambda: rand(size=(5, 7, 3, 4, 2)),
        "seed": 1,
    },
    {
        "module_fn": lambda: _initialize_training_false(BatchNorm1d(num_features=7)),
        "input_fn": lambda: rand(size=(5, 7)),
        "id_prefix": "training=False",
    },
    {
        "module_fn": lambda: _initialize_training_false(BatchNorm1d(num_features=7)),
        "input_fn": lambda: rand(size=(5, 7, 4)),
        "id_prefix": "training=False",
    },
    {
        "module_fn": lambda: _initialize_training_false(BatchNorm2d(num_features=7)),
        "input_fn": lambda: rand(size=(5, 7, 3, 4)),
        "id_prefix": "training=False",
    },
    {
        "module_fn": lambda: _initialize_training_false(BatchNorm3d(num_features=7)),
        "input_fn": lambda: rand(size=(5, 7, 3, 4, 2)),
        "id_prefix": "training=False",
    },
]
