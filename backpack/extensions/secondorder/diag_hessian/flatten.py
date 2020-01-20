from backpack.core.derivatives.flatten import FlattenDerivatives
from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule


class DiagHFlatten(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivatives())

    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        if self.is_no_op(module):
            return backproped
        else:
            return super().backpropagate(ext, module, grad_inp, grad_out, backproped)

    def is_no_op(self, module):
        """Does flatten add an operation to the computational graph.

        If the input is already flattened, no operation will be added for
        the `Flatten` layer. This can lead to an intuitive order of backward
        hook execution, see the discussion at https://discuss.pytorch.org/t/
        backward-hooks-changing-order-of-execution-in-nn-sequential/12447/4 .
        """

        return tuple(module.input0_shape) == tuple(module.output_shape)
