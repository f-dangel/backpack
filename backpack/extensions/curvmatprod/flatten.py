from backpack.core.derivatives.flatten import FlattenDerivatives

from .cmpbase import CMPBase


class CMPFlatten(CMPBase):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivatives())

    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        return backproped
