"""N-dimensional average pooling derivatives.

Average pooling can be expressed as convolution over grouped channels with a constant
kernel.
"""
from typing import Any, List, Tuple

from einops import rearrange
from torch import Tensor, ones_like
from torch.nn import Module

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.utils.conv import get_conv_module
from backpack.utils.conv_transpose import get_conv_transpose_module


class AvgPoolNDDerivatives(BaseDerivatives):
    def __init__(self, N: int):
        self.conv = get_conv_module(N)
        self.convt = get_conv_transpose_module(N)
        self.N = N

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

    def hessian_is_zero(self, module):
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
        avg_kernel = ones_like(convnd.weight) / convnd.weight.numel()
        convnd.weight.data = avg_kernel

        return convnd(mat)

    def __check_jmp_out_as_pool(self, mat, jmp_as_pool, module):
        V = mat.shape[0]
        N, C_out = module.output.shape[:2]

        assert jmp_as_pool.shape == (V * N * C_out, 1) + module.output.shape[2:]

    def _jac_t_mat_prod(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        self.check_parameters(module)

        mat_as_pool = self.__make_single_channel(mat, module)
        jmp_as_pool = self.__apply_jacobian_t_of(module, mat_as_pool)

        return self.reshape_like_input(jmp_as_pool, module, subsampling=subsampling)

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
        avg_kernel = ones_like(convnd_t.weight) / convnd_t.weight.numel()
        convnd_t.weight.data = avg_kernel

        V_N_C_in = mat.size(0)
        output_size = (V_N_C_in, C_for_conv_t) + tuple(module.input0.shape[2:])

        return convnd_t(mat, output_size=output_size)
