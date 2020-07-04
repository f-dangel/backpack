"""Partial derivatives for `torch.nn.ConvTranspose3d`."""

import torch
from torch.nn import ConvTranspose3d
from torch.nn.functional import conv3d

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils.conv_transpose import unfold_by_conv_transpose
from backpack.utils.ein import eingroup


class ConvTranspose3DDerivatives(BaseParameterDerivatives):
    def get_module(self):
        return ConvTranspose3d

    def hessian_is_zero(self):
        return True

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        N_axis, D_axis, H_axis, W_axis = 1, 3, 4, 5
        axes = [D_axis, H_axis, W_axis]
        if sum_batch:
            axes = [N_axis] + axes

        return mat.sum(axes)

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        # expand for each batch and for each channel
        N_axis, D_axis, H_axis, W_axis = 1, 3, 4, 5
        jac_mat = (
            mat.unsqueeze(N_axis).unsqueeze(D_axis).unsqueeze(H_axis).unsqueeze(W_axis)
        )

        N, _, D_out, H_out, W_out = module.output_shape
        return jac_mat.expand(-1, N, -1, D_out, H_out, W_out)

    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        V = mat.shape[0]
        G = module.groups
        C_in = module.input0.shape[1]
        _, _, K_X, K_Y, K_Z = module.weight.shape
        N, C_out, D_out, H_out, W_out = module.output.shape

        mat_reshape = mat.reshape(V, C_in, G, C_out // G, K_X, K_Y, K_Z)
        u = unfold_by_conv_transpose(module.input0, module).reshape(
            N, C_in // G, G, K_X, K_Y, K_Z, D_out, H_out, W_out
        )

        jac_mat = torch.einsum("nigxyzdhw,vigoxyz->vngodhw", u, mat_reshape)

        return self.reshape_like_output(jac_mat, module)

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        _, C_in, D_in, H_in, W_in = module.input0.size()
        in_features = C_in * D_in * H_in * W_in
        _, C_out, D_out, H_out, W_out = module.output.size()
        out_features = C_out * D_out * H_out * W_out

        mat = mat.reshape(out_features, C_out, D_out, H_out, W_out)
        jac_t_mat = self.__jac_t(module, mat).reshape(out_features, in_features)

        mat_t_jac = jac_t_mat.t().reshape(in_features, C_out, D_out, H_out, W_out)
        jac_t_mat_t_jac = self.__jac_t(module, mat_t_jac).reshape(
            in_features, in_features
        )

        return jac_t_mat_t_jac.t()

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        V = mat.shape[0]
        G = module.groups
        N, C_out, D_out, H_out, W_out = module.output.shape

        mat_reshape = mat.reshape(V, N, G, C_out // G, D_out, H_out, W_out)

        C_in = module.input0.shape[1]
        _, _, K_X, K_Y, K_Z = module.weight.shape

        u = unfold_by_conv_transpose(module.input0, module).reshape(
            N, C_in // G, G, K_X, K_Y, K_Z, D_out, H_out, W_out
        )

        result_str = "vigoxyz" if sum_batch else "vnigoxyz"
        equation = "nigxyzdhw,vngodhw->{}".format(result_str)

        final_shape = (
            (V, *module.weight.shape) if sum_batch else (V, N, *module.weight.shape)
        )

        return torch.einsum(equation, u, mat_reshape).reshape(final_shape)

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = eingroup("v,n,c,d,h,w->vn,c,d,h,w", mat)
        jmp_as_conv = self.__jac(module, mat_as_conv)
        return self.reshape_like_output(jmp_as_conv, module)

    def __jac(self, module, mat):
        C_in = module.input0.shape[1]
        _, C_out, D_out, H_out, W_out = module.output.shape
        D_axis = 2
        H_axis = 3
        W_axis = 4

        conv3d_t = ConvTranspose3d(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            bias=False,
            dilation=module.dilation,
            groups=module.groups,
        ).to(module.input0.device)

        conv3d_t.weight.data = module.weight

        V_N = mat.size(0)
        output_size = (V_N, C_out, D_out, H_out, W_out)

        jac_mat = (
            conv3d_t(mat, output_size=output_size)
            .narrow(D_axis, 0, D_out)
            .narrow(H_axis, 0, H_out)
            .narrow(W_axis, 0, W_out)
        )
        return jac_mat

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = eingroup("v,n,c,d,h,w->vn,c,d,h,w", mat)
        jmp_as_conv = self.__jac_t(module, mat_as_conv)
        return self.reshape_like_input(jmp_as_conv, module)

    def __jac_t(self, module, mat):
        """Apply ConvTranspose3d backward operation."""
        D_axis = 2
        H_axis = 3
        W_axis = 4
        D_in = module.input0.size(D_axis)
        H_in = module.input0.size(H_axis)
        W_in = module.input0.size(W_axis)

        return (
            conv3d(
                mat,
                module.weight,
                bias=None,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )
            .narrow(D_axis, 0, D_in)
            .narrow(H_axis, 0, H_in)
            .narrow(W_axis, 0, W_in)
        )
