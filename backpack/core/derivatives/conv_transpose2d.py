"""Partial derivatives for `torch.nn.ConvTranspose2d`."""

from torch.nn import ConvTranspose2d
from torch.nn.functional import conv2d

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils.ein import eingroup


class ConvTranspose2DDerivatives(BaseParameterDerivatives):
    def get_module(self):
        return ConvTranspose2d

    def hessian_is_zero(self):
        return True

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        N_axis, H_axis, W_axis = 1, 3, 4
        axes = [H_axis, W_axis]
        if sum_batch:
            axes = [N_axis] + axes

        return mat.sum(axes)

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        raise NotImplementedError

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = eingroup("v,n,c,h,w->vn,c,h,w", mat)
        jmp_as_conv = self.__jac_t(module, mat_as_conv)
        return self.reshape_like_input(jmp_as_conv, module)

    def __jac_t(self, module, mat):
        """Apply ConvTranspose2d backward operation."""
        H_axis = 2
        W_axis = 3
        H_in = module.input0.size(H_axis)
        W_in = module.input0.size(W_axis)

        return (
            conv2d(
                mat,
                module.weight,
                bias=None,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )
            .narrow(H_axis, 0, H_in)
            .narrow(W_axis, 0, W_in)
        )
