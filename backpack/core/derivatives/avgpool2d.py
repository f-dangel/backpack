"""The code relies on the insight that average pooling can be understood as
convolution over single channels with a constant kernel."""

import torch.nn
from torch.nn import AvgPool2d, Conv2d, ConvTranspose2d

from backpack.core.derivatives.utils import (
    jac_t_mat_prod_accept_vectors,
    jac_mat_prod_accept_vectors,
)

from backpack.utils.einsum import einsum, eingroup
from backpack.core.derivatives.basederivatives import BaseDerivatives


class AvgPool2DDerivatives(BaseDerivatives):
    def get_module(self):
        return AvgPool2d

    def hessian_is_zero(self):
        return True

    # TODO: Require tests
    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        """Use fact that average pooling can be implemented as conv."""
        _, channels, in_x, in_y = module.input0.size()
        in_features = channels * in_x * in_y
        _, _, out_x, out_y = module.output.size()
        out_features = channels * out_x * out_y

        # 1) apply conv_transpose to multiply with W^T
        result = mat.view(channels, out_x, out_y, out_features)
        result = einsum("cxyf->fcxy", (result,)).contiguous()
        result = result.view(out_features * channels, 1, out_x, out_y)
        # result: W^T mat
        result = self.__apply_jacobian_t_of(module, result)
        result = result.view(out_features, in_features)

        # 2) transpose: mat^T W
        result = result.t().contiguous()

        # 3) apply conv_transpose
        result = result.view(in_features * channels, 1, out_x, out_y)
        result = self.__apply_jacobian_t_of(module, result)

        # 4) transpose to obtain W^T mat W
        return result.view(in_features, in_features).t()

    def check_exotic_parameters(self, module):
        assert module.count_include_pad, (
            "Might not work for exotic hyperparameters of AvgPool2d, "
            + "like count_include_pad=False"
        )

    @jac_mat_prod_accept_vectors
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        self.check_exotic_parameters(module)

        mat_as_pool = self.__make_single_channel(mat, module)
        jmp_as_pool = self.__apply_jacobian_of(module, mat_as_pool)
        self.__check_jmp_out_as_pool(mat, jmp_as_pool, module)

        return self.__view_as_output(jmp_as_pool, module)

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

    def __view_as_output(self, mat, module):
        """Ungroup dimensions after application of Jacobian."""
        V = -1
        shape = (V, *module.output_shape)
        return mat.view(shape)

    @jac_t_mat_prod_accept_vectors
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        self.check_exotic_parameters(module)

        mat_as_pool = self.__make_single_channel(mat, module)
        jmp_as_pool = self.__apply_jacobian_t_of(module, mat_as_pool)
        self.__check_jmp_in_as_pool(mat, jmp_as_pool, module)

        return self.__view_as_input(jmp_as_pool, module)

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

    def __view_as_input(self, mat, module):
        """Ungroup dimensions after application of Jacobian."""
        V = -1
        shape = (V, *module.input0_shape)
        return mat.view(shape)
