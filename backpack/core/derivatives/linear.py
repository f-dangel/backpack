import torch
from ...utils.utils import einsum
from torch.nn import Linear
from .basederivatives import BaseDerivatives
from ..layers import LinearConcat

from .utils import unsqueeze_if_missing_dim


class LinearDerivatives(BaseDerivatives):
    def get_module(self):
        return Linear

    def get_input(self, module):
        return module.input0

    def get_batch(self, module):
        return module.input0.size(0)

    def hessian_is_zero(self):
        return True

    def get_weight_data(self, module):
        return module.weight.data

    def jac_t_mat_prod(self, module, grad_input, grad_output, mat):
        d_linear = self.get_weight_data(module)
        return einsum('ij,bic->bjc', (d_linear, mat))

    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        d_linear = self.get_weight_data(module)
        return einsum('ij,bjc->bic', (d_linear, mat))

    def ea_jac_t_mat_jac_prod(self, module, grad_input, grad_output, mat):
        jac = self.get_weight_data(module)
        return einsum('ik,ij,jl->kl', (jac, mat, jac))

    def weight_jac_mat_prod(self, module, grad_input, grad_output, mat):
        batch = self.get_batch(module)
        num_cols = mat.size(1)
        shape = tuple(module.weight.size()) + (num_cols, )

        jac_mat = einsum('bj,ijc->bic',
                         (self.get_input(module), mat.view(shape)))
        return jac_mat

    @unsqueeze_if_missing_dim(mat_dim=3)
    def weight_jac_t_mat_prod(self,
                              module,
                              grad_input,
                              grad_output,
                              mat,
                              sum_batch=True):
        batch = self.get_batch(module)
        num_cols = mat.size(2)

        equation = 'bjc,bi->jic' if sum_batch is True else 'bjc,bi->bjic'

        jac_t_mat = einsum(equation,
                           (mat, self.get_input(module))).contiguous()

        sum_shape = [module.weight.numel(), num_cols]
        shape = sum_shape if sum_batch is True else [batch] + sum_shape

        return jac_t_mat.view(*shape)

    def bias_jac_mat_prod(self, module, grad_input, grad_output, mat):
        batch = self.get_batch(module)
        return mat.unsqueeze(0).expand(batch, -1, -1)

    @unsqueeze_if_missing_dim(mat_dim=3)
    def bias_jac_t_mat_prod(self,
                            module,
                            grad_input,
                            grad_output,
                            mat,
                            sum_batch=True):
        if sum_batch is True:
            return mat.sum(0)
        else:
            return mat


class LinearConcatDerivatives(LinearDerivatives):
    # override
    def get_module(self):
        return LinearConcat

    # override
    def get_input(self, module):
        """Return homogeneous input, if bias exists """
        input = super().get_input(module)
        if module.has_bias():
            return module.append_ones(input)
        else:
            return input

    # override
    def get_weight_data(self, module):
        return module._slice_weight().data

    def bias_jac_mat_prod(self, module, grad_input, grad_output, mat):
        raise RuntimeError("Bias is concatenated with weight matrix")

    def bias_jac_t_mat_prod(self,
                            module,
                            grad_input,
                            grad_output,
                            mat,
                            sum_batch=True):
        raise RuntimeError("Bias is concatenated with weight matrix")
