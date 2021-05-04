from einops import rearrange
from numpy import prod
from torch import einsum
from torch.nn import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from torch.nn.functional import (
    conv1d,
    conv2d,
    conv3d,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
)
from torch.nn.grad import _grad_input_padding

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils.conv_transpose import unfold_by_conv_transpose


class ConvTransposeNDDerivatives(BaseParameterDerivatives):
    def __init__(self, N):
        if N == 1:
            self.module = ConvTranspose1d
            self.conv_func = conv1d
            self.conv_transpose_func = conv_transpose1d
        elif N == 2:
            self.module = ConvTranspose2d
            self.conv_func = conv2d
            self.conv_transpose_func = conv_transpose2d
        elif N == 3:
            self.module = ConvTranspose3d
            self.conv_func = conv3d
            self.conv_transpose_func = conv_transpose3d
        else:
            raise ValueError("{}-dimensional Conv. is not implemented.".format(N))
        self.conv_dims = N

    def hessian_is_zero(self):
        return True

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        axes = list(range(3, len(module.output.shape) + 1))
        if sum_batch:
            axes = [1] + axes
        return mat.sum(axes)

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        # Expand batch dimension
        jac_mat = mat.unsqueeze(1)
        # Expand data dimensions
        for i in range(3, len(module.output.shape) + 1):
            jac_mat = jac_mat.unsqueeze(i)

        expand_shape = [-1, module.output.shape[0], -1, *module.output.shape[2:]]

        return jac_mat.expand(*expand_shape)

    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        if module.groups != 1:
            raise NotImplementedError("Groups greater than 1 are not supported yet")

        V = mat.shape[0]
        G = module.groups
        C_in = module.input0.shape[1]
        N = module.output.shape[0]
        C_out = module.output.shape[1]

        mat_reshape = mat.reshape(V, C_in, G, C_out // G, *module.weight.shape[2:])
        u = unfold_by_conv_transpose(module.input0, module).reshape(
            N, C_in // G, G, *module.weight.shape[2:], *module.output.shape[2:]
        )

        dims_kern = "xyz"[: self.conv_dims]
        dims_data = "abc"[: self.conv_dims]
        einstr = "nig{0}{1},vigo{0}->vngo{1}".format(dims_kern, dims_data)
        jac_mat = einsum(einstr, u, mat_reshape)

        return self.reshape_like_output(jac_mat, module)

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Apply transposed Jacobian of the output w.r.t.
        weight of the torch.nn.ConvTransposeNd layer to a matrix.

        Parameters:
        -----------
        mat: torch.Tensor
            Matrix the transposed Jacobian will be applied to.
            Must have shape [V, N, C_out, H_out, ...].
        sum_batch: bool
            Whether to sum over the batch dimension on the fly.

        Returns:
        --------
        result: torch.Tensor
            Jacobian-matrix product.
            Has shape [V, N, C_w, H_w, ...] if `sum_batch == False`.
            Has shape [V, C_w, H_w, ...] if `sum_batch == True`."""
        V = mat.shape[0]
        G = module.groups
        C_in = module.input0.shape[1]
        N = module.output.shape[0]
        C_out = module.output.shape[1]

        mat_reshape = mat.reshape(V, N, G, C_out // G, *module.output.shape[2:])

        u = unfold_by_conv_transpose(module.input0, module).reshape(
            N, G, C_in // G, *module.weight.shape[2:], *module.output.shape[2:]
        )

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

        jac_t_mat = conv_transpose1d(
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

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = rearrange(mat, "v n c ... -> (v n) c ...")
        jmp_as_conv = self.__jac_t(module, mat_as_conv)
        return self.reshape_like_input(jmp_as_conv, module)

    def __jac_t(self, module, mat):
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
