from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.module_extension import ModuleExtension


class FirstOrderModuleExtension(ModuleExtension):
    def backpropagate(self, ext, module, g_inp, g_out, bpQuantities):
        return None


class FirstOrderBackpropExtension(BackpropExtension):
    """Base backpropagation extension for first order."""

    def expects_backpropagation_quantities(self) -> bool:  # noqa: D102
        return False
