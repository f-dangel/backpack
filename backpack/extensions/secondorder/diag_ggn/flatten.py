from backpack.core.derivatives.flatten import FlattenDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNFlatten(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivatives())

    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        if self.derivatives.is_no_op(module):
            return backproped
        else:
            return super().backpropagate(ext, module, grad_inp, grad_out, backproped)
