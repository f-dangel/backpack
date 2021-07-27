"""Base class for first order extensions."""
from typing import Dict, Type

from torch.nn import Module

from backpack.extensions.backprop_extension import FAIL_WARN, BackpropExtension
from backpack.extensions.module_extension import ModuleExtension


class FirstOrderModuleExtension(ModuleExtension):
    """Base class for first order module extensions."""

    def backpropagate(self, ext, module, g_inp, g_out, bpQuantities):  # noqa: D102
        pass


class FirstOrderBackpropExtension(BackpropExtension):
    """Base backpropagation extension for first order."""

    def __init__(
        self, savefield: str, module_exts: Dict[Type[Module], ModuleExtension]
    ):  # noqa: D107
        super().__init__(savefield, module_exts, fail_mode=FAIL_WARN)

    def expects_backpropagation_quantities(self) -> bool:  # noqa: D102
        return False
