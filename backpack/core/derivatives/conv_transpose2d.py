"""Partial derivatives for `torch.nn.ConvTranspose2d`."""

import torch
from torch.nn import ConvTranspose2d
from torch.nn.functional import conv2d, conv_transpose2d

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils.conv_transpose import unfold_by_conv_transpose
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

    def bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        # TODO Implement with unfold
        raise NotImplementedError

    def weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        # TODO Implement with unfold
        raise NotImplementedError

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Unintuitive, but faster due to convolution.

        For 1d transpose convolution (x, W) ↦ y with u = unfold(x):

        y[N,G,O,H] = ∑_{I,X} W[I,G,O,X] u[N,I,G,X,H]

        ∇W[n,i,g,o,x] = ∑_{G,O,H} ∂y[n,G,O,H]/∂W[i,g,o,x] ∇y[n,G,O,H]
                      = ∑_{G,O,H,I,X} ∂(W[I,G,O,X] u[n,I,G,X,H])/∂W[i,g,o,x] ∇y[n,G,O,H]
                      = ∑_{H} u[n,i,g,x,H]) ∇y[n,g,o,H]
        """
        V = mat.shape[0]
        G = module.groups
        N, C_out, H_out, W_out = module.output.shape

        mat_reshape = mat.reshape(V, N, G, C_out // G, H_out, W_out)

        C_in = module.input0.shape[1]
        _, _, K_X, K_Y = module.weight.shape

        u = unfold_by_conv_transpose(module.input0, module).reshape(
            N, C_in // G, G, K_X, K_Y, H_out, W_out
        )

        result_str = "vigoxy" if sum_batch else "vnigoxy"
        equation = "nigxyhw,vngohw->{}".format(result_str)

        final_shape = (
            (V, *module.weight.shape) if sum_batch else (V, N, *module.weight.shape)
        )

        return torch.einsum(equation, u, mat_reshape).reshape(final_shape)

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