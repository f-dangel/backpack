from .basederivatives import BaseDerivatives
from ...utils.utils import einsum

from .utils import jmp_unsqueeze_if_missing_dim, hmp_unsqueeze_if_missing_dim


class ElementwiseDerivatives(BaseDerivatives):
    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def jac_t_mat_prod(self, module, grad_input, grad_output, mat):
        _, df_flat = self.batch_flat(self.df(module, grad_input, grad_output))
        return einsum('bi,bic->bic', (df_flat, mat))

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        _, df_flat = self.batch_flat(self.df(module, grad_input, grad_output))
        return einsum('bi,bic->bic', (df_flat, mat))

    def ea_jac_t_mat_jac_prod(self, module, grad_input, grad_output, mat):
        batch, df_flat = self.batch_flat(
            self.df(module, grad_input, grad_output))
        return einsum('bi,ij,bj->ij', (df_flat, mat, df_flat)) / batch

    def hessian_diagonal(self, module, grad_input, grad_output):
        _, d2f_flat = self.batch_flat(
            self.d2f(module, grad_input, grad_output))
        _, grad_output_flat = self.batch_flat(grad_output[0])
        return d2f_flat * grad_output_flat

    def df(self, module, grad_input, grad_output):
        raise NotImplementedError("First derivatives not implemented")

    def d2f(self, module, grad_input, grad_output):
        raise NotImplementedError("Second derivatives not implemented")
