"""The code relies on the insight that average pooling can be understood as
convolution over single channels with a constant kernel."""

import torch.nn
from torch.nn import Conv2d, ConvTranspose2d

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.utils.ein import eingroup


class AvgPool2DDerivatives(BaseDerivatives):
    def hessian_is_zero(self):
        return True

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        """Use fact that average pooling can be implemented as conv."""
        _, C, H_in, W_in = module.input0.size()
        in_features = C * H_in * W_in
        _, _, H_out, W_out = module.output.size()
        out_features = C * H_out * W_out

        mat = mat.reshape(out_features * C, 1, H_out, W_out)
        jac_t_mat = self.__apply_jacobian_t_of(module, mat).reshape(
            out_features, in_features
        )
        mat_t_jac = jac_t_mat.t().reshape(in_features * C, 1, H_out, W_out)
        jac_t_mat_t_jac = self.__apply_jacobian_t_of(module, mat_t_jac).reshape(
            in_features, in_features
        )

        return jac_t_mat_t_jac.t()

    def check_exotic_parameters(self, module):
        assert module.count_include_pad, (
            "Might not work for exotic hyperparameters of AvgPool2d, "
            + "like count_include_pad=False"
        )

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        self.check_exotic_parameters(module)

        mat_as_pool = self.__make_single_channel(mat, module)
        jmp_as_pool = self.__apply_jacobian_of(module, mat_as_pool)
        self.__check_jmp_out_as_pool(mat, jmp_as_pool, module)

        return self.reshape_like_output(jmp_as_pool, module)

    def __make_single_channel(self, mat, module):
        """Create fake single-channel images, grouping batch,
        class and channel dimension."""
        result = eingroup("v,n,c,w,h->vnc,w,h", mat)
        C_axis = 1
        return result.unsqueeze(C_axis)

    def __apply_jacobian_of(self, module, mat):
        conv2d = Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            bias=False,
        ).to(module.input0.device)

        conv2d.weight.requires_grad = False
        avg_kernel = torch.ones_like(conv2d.weight) / conv2d.weight.numel()
        conv2d.weight.data = avg_kernel

        return conv2d(mat)

    def __check_jmp_out_as_pool(self, mat, jmp_as_pool, module):
        V = mat.size(0)
        N, C_out, H_out, W_out = module.output_shape
        assert jmp_as_pool.shape == (V * N * C_out, 1, H_out, W_out)

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        self.check_exotic_parameters(module)

        mat_as_pool = self.__make_single_channel(mat, module)
        jmp_as_pool = self.__apply_jacobian_t_of(module, mat_as_pool)
        self.__check_jmp_in_as_pool(mat, jmp_as_pool, module)

        return self.reshape_like_input(jmp_as_pool, module)

    def __apply_jacobian_t_of(self, module, mat):
        C_for_conv_t = 1

        conv2d_t = ConvTranspose2d(
            in_channels=C_for_conv_t,
            out_channels=C_for_conv_t,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            bias=False,
        ).to(module.input0.device)

        conv2d_t.weight.requires_grad = False
        avg_kernel = torch.ones_like(conv2d_t.weight) / conv2d_t.weight.numel()
        conv2d_t.weight.data = avg_kernel

        V_N_C_in = mat.size(0)
        _, _, H_in, W_in = module.input0.size()
        output_size = (V_N_C_in, C_for_conv_t, H_in, W_in)

        return conv2d_t(mat, output_size=output_size)

    def __check_jmp_in_as_pool(self, mat, jmp_as_pool, module):
        V = mat.size(0)
        N, C_in, H_in, W_in = module.input0_shape
        assert jmp_as_pool.shape == (V * N * C_in, 1, H_in, W_in)
