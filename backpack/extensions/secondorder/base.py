"""Contains base classes for second order extensions."""
import warnings

from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.backprop_extension import FAIL_ERROR, FAIL_SILENT, FAIL_WARN
from backpack.utils.errors import change_error_to_warn_message
from torch.nn import Module


class SecondOrderBackpropExtension(BackpropExtension):
    """Base backpropagation extension for second order."""

    def _handle_missing_module_extension(self, module: Module) -> None:
        message = (
            f"Extension saving to {self.savefield} "
            f"does not have an extension for Module {module.__class__}. "
            "Further computations will likely fail as second-order quantities "
            "will not be backpropagated."
        )

        if self._fail_mode is FAIL_ERROR:
            # PyTorch converts this Error into a RuntimeError for torch<1.7.0
            raise NotImplementedError(message + " " + change_error_to_warn_message)
        elif self._fail_mode == FAIL_WARN:
            warnings.warn(message)

    def expects_backpropagation_quantities(self) -> bool:  # noqa: D102
        return True
