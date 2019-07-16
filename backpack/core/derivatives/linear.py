from ...utils.utils import einsum
from torch.nn import Linear
from .basederivatives import BaseDerivatives


class LinearDerivatives(BaseDerivatives):
    def get_module(self):
        return Linear

    def hessian_is_zero(self):
        return True

    def jac_t_mat_prod(self, module, grad_input, grad_output, mat):
        d_linear = module.weight.data
        return einsum('ij,bic->bjc', (d_linear, mat))

    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        d_linear = module.weight.data
        return einsum('ij,bjc->bic', (d_linear, mat))

    def ea_jac_t_mat_jac_prod(self, module, grad_input, grad_output, mat):
        jac = module.weight
        return einsum('ik,ij,jl->kl', (jac, mat, jac))

    def weight_jac_mat_prod(self, module, grad_input, grad_output, mat):
        batch = module.input0.size(0)
        num_cols = mat.size(1)
        shape = tuple(module.weight.size()) + (num_cols,)

        jac_mat = einsum('bj,ijc->bic', (module.input0, mat.view(shape)))
        return jac_mat

    def weight_jac_t_mat_prod(self, module, grad_input, grad_output, mat):
        batch = module.input0.size(0)
        num_cols = mat.size(2)

        jac_t_mat = einsum('bjc,bi->jic', (mat, module.input0)).contiguous()
        return jac_t_mat.view(module.weight.numel(), num_cols)

    def bias_jac_mat_prod(self, module, grad_input, grad_output, mat):
        batch = module.input0.size(0)
        return mat.unsqueeze(0).expand(batch, -1, -1)

    def bias_jac_t_mat_prod(self, module, grad_input, grad_output, mat):
        return mat.sum(0)
