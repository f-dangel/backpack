"""Partial derivatives for ``torch.nn.ConvTranspose{1,2,3}d``."""
from typing import List, Tuple, Union

from einops import rearrange
from numpy import prod
from torch import Tensor, einsum
from torch.nn import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Module
from torch.nn.grad import _grad_input_padding

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils.conv import get_conv_function
from backpack.utils.conv_transpose import (
    get_conv_transpose_function,
    unfold_by_conv_transpose,
)
from backpack.utils.subsampling import subsample


class ConvTransposeNDDerivatives(BaseParameterDerivatives):
    """Base class for partial derivatives of transpose convolution."""

    def __init__(self, N: int):
        """Store transpose convolution dimension and operations.

        Args:
            N: Transpose convolution dimension.
        """
        self.conv_func = get_conv_function(N)
        self.conv_transpose_func = get_conv_transpose_function(N)
        self.conv_dims = N

    def hessian_is_zero(self, module):
        return True

    def _bias_jac_t_mat_prod(
        self, module, g_inp, g_out, mat, sum_batch=True, subsampling=None
    ):
        equation = f"vnc...->v{'' if sum_batch else 'n'}c"
        return einsum(equation, mat)

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        # Expand batch dimension
        jac_mat = mat.unsqueeze(1)
        # Expand data dimensions
        for i in range(3, len(module.output.shape) + 1):
            jac_mat = jac_mat.unsqueeze(i)

        expand_shape = [-1, module.output.shape[0], -1, *module.output.shape[2:]]

        return jac_mat.expand(*expand_shape)

    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        V = mat.shape[0]
        G = module.groups
        C_in = module.input0.shape[1]
        N = module.output.shape[0]
        C_out = module.output.shape[1]

        mat_reshape = mat.reshape(V, G, C_in // G, C_out // G, *module.weight.shape[2:])
        u = unfold_by_conv_transpose(module.input0, module).reshape(
            N, G, C_in // G, *module.weight.shape[2:], *module.output.shape[2:]
        )

        dims_kern = "xyz"[: self.conv_dims]
        dims_data = "abc"[: self.conv_dims]
        einstr = "ngi{0}{1},vgio{0}->vngo{1}".format(dims_kern, dims_data)
        jac_mat = einsum(einstr, u, mat_reshape)

        return self.reshape_like_output(jac_mat, module)

    def _weight_jac_t_mat_prod(
        self,
        module: Union[ConvTranspose1d, ConvTranspose2d, ConvTranspose3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        V = mat.shape[0]
        G = module.groups
        C_in = module.input0.shape[1]
        N = module.output.shape[0] if subsampling is None else len(subsampling)
        C_out = module.output.shape[1]

        mat_reshape = mat.reshape(V, N, G, C_out // G, *module.output.shape[2:])

        u = unfold_by_conv_transpose(
            subsample(module.input0, subsampling=subsampling), module
        ).reshape(N, G, C_in // G, *module.weight.shape[2:], *module.output.shape[2:])

        dims_kern = "xyz"[: self.conv_dims]
        dims_data = "abc"[: self.conv_dims]
        result_str = ("vgio" if sum_batch else "vngio") + dims_kern
        equation = f"ngi{dims_kern}{dims_data},vngo{dims_data}->{result_str}"

        final_shape = (
            (V, *module.weight.shape) if sum_batch else (V, N, *module.weight.shape)
        )

        return einsum(equation, u, mat_reshape).reshape(final_shape)

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        in_features = int(prod(module.input0.size()[1:]))
        out_features = int(prod(module.output.size()[1:]))

        mat = mat.reshape(out_features, *module.output.size()[1:])
        jac_t_mat = self.__jac_t(module, mat).reshape(out_features, in_features)

        mat_t_jac = jac_t_mat.t().reshape(in_features, *module.output.size()[1:])
        jac_t_mat_t_jac = self.__jac_t(module, mat_t_jac)
        jac_t_mat_t_jac = jac_t_mat_t_jac.reshape(in_features, in_features)

        return jac_t_mat_t_jac.t()

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = rearrange(mat, "v n c ... -> (v n) c ...")
        jmp_as_conv = self.__jac(module, mat_as_conv)
        return self.reshape_like_output(jmp_as_conv, module)

    def __jac(self, module, mat):
        input_size = list(module.output.size())
        input_size[0] = mat.size(0)

        grad_padding = _grad_input_padding(
            grad_output=mat,
            input_size=input_size,
            stride=module.stride,
            padding=module.padding,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
        )

        jac_t_mat = self.conv_transpose_func(
            input=mat,
            weight=module.weight,
            bias=None,
            stride=module.stride,
            padding=module.padding,
            output_padding=grad_padding,
            groups=module.groups,
            dilation=module.dilation,
        )

        return jac_t_mat

    def _jac_t_mat_prod(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        mat_as_conv = rearrange(mat, "v n c ... -> (v n) c ...")
        jmp_as_conv = self.__jac_t(module, mat_as_conv)
        return self.reshape_like_input(jmp_as_conv, module, subsampling=subsampling)

    def __jac_t(self, module: Module, mat: Tensor) -> Tensor:
        jac_t = self.conv_func(
            mat,
            module.weight,
            bias=None,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

        for dim in range(self.conv_dims):
            axis = dim + 1
            size = module.input0.shape[axis]
            jac_t = jac_t.narrow(axis, 0, size)

        return jac_t
