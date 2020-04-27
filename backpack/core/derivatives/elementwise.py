from torch import einsum

from backpack.core.derivatives.basederivatives import BaseDerivatives


class ElementwiseDerivatives(BaseDerivatives):
    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        self._no_inplace(module)

        df_elementwise = self.df(module, g_inp, g_out)
        return einsum("...,v...->v...", (df_elementwise, mat))

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        self._no_inplace(module)

        return self.jac_t_mat_prod(module, g_inp, g_out, mat)

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        self._no_inplace(module)

        batch, df_flat = self.batch_flat(self.df(module, g_inp, g_out))
        return einsum("ni,nj,ij->ij", (df_flat, df_flat, mat)) / batch

    def hessian_diagonal(self, module, g_inp, g_out):
        self._no_inplace(module)

        return self.d2f(module, g_inp, g_out) * g_out[0]

    def df(self, module, g_inp, g_out):
        raise NotImplementedError("First derivatives not implemented")

    def d2f(self, module, g_inp, g_out):
        raise NotImplementedError("Second derivatives not implemented")

    @staticmethod
    def _no_inplace(module):
        """Do not support inplace modification.

        Jacobians/Hessians might be computed using the modified input instead
        of the original.

        Args:
            module (torch.nn.Module): Elementwise activation module.

        Raises:
            NotImplementedError: If `module` has inplace option enabled.

        Todo:
            - Write tests to investigate what happens with `inplace=True`.
        """
        has_inplace_option = hasattr(module, "inplace")

        if has_inplace_option:
            if module.inplace is True:
                raise NotImplementedError("Inplace not supported in {}.".format(module))
