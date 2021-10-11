"""Base class for first order extensions."""
from typing import Dict, List, Type

from torch.nn import Module

from backpack.extensions.backprop_extension import FAIL_WARN, BackpropExtension
from backpack.extensions.module_extension import ModuleExtension


class FirstOrderModuleExtension(ModuleExtension):
    """Base class for first order module extensions."""


class FirstOrderBackpropExtension(BackpropExtension):
    """Base backpropagation extension for first order."""

    def __init__(
        self,
        savefield: str,
        module_exts: Dict[Type[Module], ModuleExtension],
        fail_mode: str = FAIL_WARN,
        subsampling: List[int] = None,
    ):  # noqa: D107
        super().__init__(
            savefield, module_exts, fail_mode=fail_mode, subsampling=subsampling
        )

    def expects_backpropagation_quantities(self) -> bool:  # noqa: D102
        return False
