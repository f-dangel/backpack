from backpack.core.derivatives.flatten import FlattenDerivatives

from .diag_h_base import DiagHBaseModule


class DiagHFlatten(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivatives())

    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        return backproped
