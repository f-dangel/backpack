from backpack.core.derivatives.flatten import FlattenDerivatives

from .hbpbase import HBPBaseModule


class HBPFlatten(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivatives())

    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        if self.is_no_op(module):
            return backproped
        else:
            return super().backpropagate(ext, module, grad_inp, grad_out, backproped)

    def is_no_op(self, module):
        return tuple(module.input0_shape) == tuple(module.output_shape)
