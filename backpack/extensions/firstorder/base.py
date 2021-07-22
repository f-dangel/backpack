"""Base class for first order extensions."""
from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.module_extension import ModuleExtension


class FirstOrderModuleExtension(ModuleExtension):
    """Base class for first order module extensions."""

    def backpropagate(self, ext, module, g_inp, g_out, bpQuantities):  # noqa: D102
        pass


class FirstOrderBackpropExtension(BackpropExtension):
    """Base backpropagation extension for first order."""

    def expects_backpropagation_quantities(self) -> bool:  # noqa: D102
        return False
