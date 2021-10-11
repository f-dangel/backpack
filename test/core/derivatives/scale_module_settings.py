"""Test settings for ScaleModule derivatives."""
from torch import rand
from torch.nn import Identity

from backpack.custom_module.scale_module import ScaleModule

SCALE_MODULE_SETTINGS = [
    {
        "module_fn": lambda: ScaleModule(),
        "input_fn": lambda: rand(3, 4, 2),
    },
    {
        "module_fn": lambda: ScaleModule(0.3),
        "input_fn": lambda: rand(3, 2),
    },
    {
        "module_fn": lambda: ScaleModule(5.7),
        "input_fn": lambda: rand(2, 3),
    },
    {
        "module_fn": lambda: Identity(),
        "input_fn": lambda: rand(3, 1, 2),
    },
]
