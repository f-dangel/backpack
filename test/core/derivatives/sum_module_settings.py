"""Test settings for SumModule derivatives."""
from torch import rand

from backpack.custom_module.branching import SumModule

SUM_MODULE_SETTINGS = [
    {
        "module_fn": lambda: SumModule(),
        "input_fn": lambda: (rand(4, 3), rand(4, 3)),
    },
    {
        "module_fn": lambda: SumModule(),
        "input_fn": lambda: (rand(2, 3), rand(2, 3), rand(2, 3)),
    },
]
