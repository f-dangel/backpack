"""N-dimensional average pooling derivatives.

Average pooling can be expressed as convolution over grouped channels with a constant
kernel.
"""
from typing import Any, Tuple

import torch.nn
from einops import rearrange
from torch.nn import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    Module,
)

from backpack.core.derivatives.basederivatives import BaseDerivatives


class AvgPoolNDDerivatives(BaseDerivatives):
    def __init__(self, N):
        self.N = N
        if self.N == 1:
            self.conv = Conv1d
            self.convt = ConvTranspose1d
        elif self.N == 2:
            self.conv = Conv2d
            self.convt = ConvTranspose2d
        elif self.N == 3:
            self.conv = Conv3d
            self.convt = ConvTranspose3d

    def check_parameters(self, module: Module) -> None:
        assert module.count_include_pad, (
            "Might not work for exotic hyperparameters of AvgPool2d, "
            + "like count_include_pad=False"
        )

    def get_avg_pool_parameters(self, module) -> Tuple[Any, Any, Any]:
        """Return the parameters of the module.

        Args:
            module: module

        Returns:
            stride, kernel_size, padding
        """
        return module.stride, module.kernel_size, module.padding

    def hessian_is_zero(self):
        return True

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        """Use fact that average pooling can be implemented as conv."""
        self.check_parameters(module)

        C = module.input0.shape[1]
        shape_out = (1,) + tuple(module.output.shape[2:])
        in_features = module.input0.shape[1:].numel()
        out_features = module.output.shape[1:].numel()

        mat = mat.reshape(out_features * C, *shape_out)
        jac_t_mat = self.__apply_jacobian_t_of(module, mat).reshape(
            out_features, in_features
        )
        mat_t_jac = jac_t_mat.t().reshape(in_features * C, *shape_out)
        jac_t_mat_t_jac = self.__apply_jacobian_t_of(module, mat_t_jac).reshape(
            in_features, in_features
        )

        return jac_t_mat_t_jac.t()

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        self.check_parameters(module)

        mat_as_pool = self.__make_single_channel(mat, module)
        jmp_as_pool = self.__apply_jacobian_of(module, mat_as_pool)
        self.__check_jmp_out_as_pool(mat, jmp_as_pool, module)

        return self.reshape_like_output(jmp_as_pool, module)

    def __make_single_channel(self, mat, module):
        """Create fake single-channel images, grouping batch,
        class and channel dimension."""
        result = rearrange(mat, "v n c ... -> (v n c) ...")
        C_axis = 1
        return result.unsqueeze(C_axis)

    def __apply_jacobian_of(self, module, mat):
        stride, kernel_size, padding = self.get_avg_pool_parameters(module)
        convnd = self.conv(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ).to(module.input0.device)

        convnd.weight.requires_grad = False
        avg_kernel = torch.ones_like(convnd.weight) / convnd.weight.numel()
        convnd.weight.data = avg_kernel

        return convnd(mat)

    def __check_jmp_out_as_pool(self, mat, jmp_as_pool, module):
        V = mat.size(0)
        if self.N == 1:
            N, C_out, L_out = module.output.shape
            assert jmp_as_pool.shape == (V * N * C_out, 1, L_out)
        elif self.N == 2:
            N, C_out, H_out, W_out = module.output.shape
            assert jmp_as_pool.shape == (V * N * C_out, 1, H_out, W_out)
        elif self.N == 3:
            N, C_out, D_out, H_out, W_out = module.output.shape
            assert jmp_as_pool.shape == (V * N * C_out, 1, D_out, H_out, W_out)

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        self.check_parameters(module)

        mat_as_pool = self.__make_single_channel(mat, module)
        jmp_as_pool = self.__apply_jacobian_t_of(module, mat_as_pool)
        self.__check_jmp_in_as_pool(mat, jmp_as_pool, module)

        return self.reshape_like_input(jmp_as_pool, module)

    def __apply_jacobian_t_of(self, module, mat):
        stride, kernel_size, padding = self.get_avg_pool_parameters(module)
        C_for_conv_t = 1

        convnd_t = self.convt(
            in_channels=C_for_conv_t,
            out_channels=C_for_conv_t,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ).to(module.input0.device)

        convnd_t.weight.requires_grad = False
        avg_kernel = torch.ones_like(convnd_t.weight) / convnd_t.weight.numel()
        convnd_t.weight.data = avg_kernel

        V_N_C_in = mat.size(0)
        output_size = (V_N_C_in, C_for_conv_t) + tuple(module.input0.shape[2:])

        return convnd_t(mat, output_size=output_size)

    def __check_jmp_in_as_pool(self, mat, jmp_as_pool, module):
        V = mat.size(0)
        if self.N == 1:
            N, C_in, L_in = module.input0.size()
            assert jmp_as_pool.shape == (V * N * C_in, 1, L_in)
        elif self.N == 2:
            N, C_in, H_in, W_in = module.input0.size()
            assert jmp_as_pool.shape == (V * N * C_in, 1, H_in, W_in)
        elif self.N == 3:
            N, C_in, D_in, H_in, W_in = module.input0.size()
            assert jmp_as_pool.shape == (V * N * C_in, 1, D_in, H_in, W_in)
