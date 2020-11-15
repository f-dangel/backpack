from backpack.core.derivatives.basederivatives import BaseDerivatives


class FlattenDerivatives(BaseDerivatives):
    def hessian_is_zero(self):
        return True

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        return mat

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        return self.reshape_like_input(mat, module)

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        return self.reshape_like_output(mat, module)

    def is_no_op(self, module):
        """Does flatten add an operation to the computational graph.

        If the input is already flattened, no operation will be added for
        the `Flatten` layer. This can lead to an intuitive order of backward
        hook execution, see the discussion at https://discuss.pytorch.org/t/
        backward-hooks-changing-order-of-execution-in-nn-sequential/12447/4 .
        """
        return tuple(module.input0_shape) == tuple(module.output_shape)
