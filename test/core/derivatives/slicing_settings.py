"""Contains test cases of BackPACK's custom Slicing module."""


from torch import rand

from backpack.custom_module.slicing import Slicing

CUSTOM_SLICING_SETTINGS = [
    {
        "module_fn": lambda: Slicing((slice(None), 0)),
        "input_fn": lambda: rand(size=(2, 4, 2, 5)),
    },
    {
        "module_fn": lambda: Slicing((slice(None),)),
        "input_fn": lambda: rand(size=(3, 4, 2, 5)),
    },
    {
        "module_fn": lambda: Slicing((slice(None), 2)),
        "input_fn": lambda: rand(size=(3, 4, 2, 5)),
    },
    {
        "module_fn": lambda: Slicing((slice(None), 2, slice(1, 2), slice(0, 5, 2))),
        "input_fn": lambda: rand(size=(3, 4, 2, 5)),
    },
]
