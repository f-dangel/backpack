"""Base class for first order extensions."""
import warnings
from typing import Dict, List, Type

from backpack.utils.errors import change_error_to_warn_message
from torch.nn import Module

from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.backprop_extension import FAIL_ERROR, FAIL_WARN
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

    def _handle_missing_module_extension(self, module: Module) -> None:
        message = (
            f"Extension saving to {self.savefield} "
            f"does not have an extension for Module {module.__class__}, "
            "but it has parameters. "
            f"Those parameters will not have their field {self.savefield} set."
        )

        for _ in module.parameters():
            if self._fail_mode is FAIL_ERROR:
                # PyTorch converts this Error into a RuntimeError for torch<1.7.0
                raise NotImplementedError(message + " " + change_error_to_warn_message)
            elif self._fail_mode == FAIL_WARN:
                warnings.warn(message)
            break
