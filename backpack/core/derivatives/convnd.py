import warnings

from einops import rearrange, reduce
from numpy import prod
from torch import einsum
from torch.nn import Conv1d, Conv2d, Conv3d
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
from backpack.utils import conv as convUtils


class weight_jac_t_save_memory:
    """Choose algorithm to apply transposed convolution weight Jacobian."""

    _SAVE_MEMORY = False

    def __init__(self, save_memory=True):
        self._new_save_memory = save_memory

    def __enter__(self):
        """Store current value, set new value."""
        self._old_save_memory = weight_jac_t_save_memory._SAVE_MEMORY
        weight_jac_t_save_memory._SAVE_MEMORY = self._new_save_memory

    def __exit__(self, type, value, traceback):
        """Restore original value."""
        weight_jac_t_save_memory._SAVE_MEMORY = self._old_save_memory


class ConvNDDerivatives(BaseParameterDerivatives):
    def __init__(self, N):
        if N == 1:
            self.module = Conv1d
            self.conv_func = conv1d
            self.conv_transpose_func = conv_transpose1d
        elif N == 2:
            self.module = Conv2d
            self.conv_func = conv2d
            self.conv_transpose_func = conv_transpose2d
        elif N == 3:
            self.module = Conv3d
            self.conv_func = conv3d
            self.conv_transpose_func = conv_transpose3d
        else:
            raise ValueError("{}-dimensional Conv. is not implemented.".format(N))
        self.conv_dims = N

    def hessian_is_zero(self):
        return True

    def get_unfolded_input(self, module):
        return convUtils.unfold_by_conv(module.input0, module)

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = rearrange(mat, "v n c ... -> (v n) c ...")
        jmp_as_conv = self.conv_func(
            mat_as_conv,
            module.weight.data,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        return self.reshape_like_output(jmp_as_conv, module)

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = rearrange(mat, "v n c ... -> (v n) c ...")
        jmp_as_conv = self.__jac_t(module, mat_as_conv)
        return self.reshape_like_input(jmp_as_conv, module)

    def __jac_t(self, module, mat):
        input_size = list(module.input0.size())
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

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        """mat has shape [V, C_out]"""
        # Expand batch dimension
        jac_mat = mat.unsqueeze(1)
        # Expand data dimensions
        for i in range(3, len(module.output.shape) + 1):
            jac_mat = jac_mat.unsqueeze(i)

        expand_shape = [-1, module.output.shape[0], -1, *module.output.shape[2:]]

        return jac_mat.expand(*expand_shape)

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        axes = list(range(3, len(module.output.shape) + 1))
        if sum_batch:
            axes = [1] + axes
        return mat.sum(axes)

    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        # separate output channel groups
        jac_mat = rearrange(mat, "v (g o) i ... -> v g o (i ...)", g=module.groups)

        X = self.get_unfolded_input(module)
        # separate input channel groups
        X = rearrange(X, "n (g i) j -> n g i j", g=module.groups)
        jac_mat = einsum("ngij,vgki->vngkj", X, jac_mat)

        return self.reshape_like_output(jac_mat, module)

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        save_memory = weight_jac_t_save_memory._SAVE_MEMORY

        if save_memory and self.conv_dims in [1, 2]:
            return self.__higher_conv_weight_jac_t(module, mat, sum_batch)

        else:

            if save_memory and self.conv_dims == 3:
                warnings.warn(
                    UserWarning(
                        "Conv3d: Cannot save memory as there is no Conv4d."
                        + " Fallback to more memory-intense method."
                    )
                )

            return self.__same_conv_weight_jac_t(module, mat, sum_batch)

    def __same_conv_weight_jac_t(self, module, mat, sum_batch):
        """Uses convolution of same order."""
        G = module.groups
        V = mat.shape[0]
        N, C_out = module.output.shape[0], module.output.shape[1]
        C_in = module.input0.shape[1]
        C_in_axis = 1
        N_axis = 0

        # treat channel groups like vectorization (v) and batch (n) axes
        mat = rearrange(mat, "v n (g c) ... -> (v n g) c ...", g=G, c=C_out // G)
        repeat_pattern = [1, C_in // G] + [1 for _ in range(self.conv_dims)]
        mat = mat.repeat(*repeat_pattern)
        mat = rearrange(mat, "a b ... -> (a b) ...")
        mat = mat.unsqueeze(C_in_axis)

        input = rearrange(module.input0, "n c ... -> (n c) ...")
        input = input.unsqueeze(N_axis)
        repeat_pattern = [1, V] + [1 for _ in range(self.conv_dims)]
        input = input.repeat(*repeat_pattern)

        grad_weight = self.conv_func(
            input,
            mat,
            bias=None,
            stride=module.dilation,
            padding=module.padding,
            dilation=module.stride,
            groups=C_in * N * V,
        ).squeeze(0)

        for dim in range(self.conv_dims):
            axis = dim + 1
            size = module.weight.shape[2 + dim]
            grad_weight = grad_weight.narrow(axis, 0, size)

        dim = {"g": G, "v": V, "n": N, "i": C_in // G, "o": C_out // G}
        if sum_batch:
            return reduce(grad_weight, "(v n g i o) ... -> v (g o) i ...", "sum", **dim)
        else:
            return rearrange(grad_weight, "(v n g i o) ... -> v n (g o) i ...", **dim)

    def __higher_conv_weight_jac_t(self, module, mat, sum_batch):
        """Requires higher-order convolution.

        The algorithm is proposed in:

            - Rochette, G., Manoel, A., & Tramel, E. W., Efficient per-example
              gradient computations in convolutional neural networks (2019).
        """
        G = module.groups
        V = mat.shape[0]
        N, C_out = module.output.shape[0], module.output.shape[1]
        C_in = module.input0.shape[1]

        if self.conv_dims == 1:
            _, _, L_in = module.input0.size()
            higher_conv_func = conv2d
            K_L_axis = 2
            K_L = module.kernel_size[0]
            spatial_dim = (C_in // G, L_in)
            spatial_dim_axis = (1, V, 1, 1)
            spatial_dim_new = (C_in // G, K_L)
        else:
            _, _, H_in, W_in = module.input0.size()
            higher_conv_func = conv3d
            K_H_axis, K_W_axis = 2, 3
            K_H, K_W = module.kernel_size
            spatial_dim = (C_in // G, H_in, W_in)
            spatial_dim_axis = (1, V, 1, 1, 1)
            spatial_dim_new = (C_in // G, K_H, K_W)

        # Reshape to extract groups from the convolutional layer
        # Channels are seen as an extra spatial dimension with kernel size 1
        input_conv = module.input0.reshape(1, N * G, *spatial_dim).repeat(
            *spatial_dim_axis
        )
        # Compute convolution between input and output; the batchsize is seen
        # as channels, taking advantage of the `groups` argument
        mat_conv = rearrange(mat, "v n c ... -> (v n c) ...").unsqueeze(1).unsqueeze(2)

        stride = (1, *module.stride)
        dilation = (1, *module.dilation)
        padding = (0, *module.padding)

        conv = higher_conv_func(
            input_conv,
            mat_conv,
            groups=V * N * G,
            stride=dilation,
            dilation=stride,
            padding=padding,
        ).squeeze(0)

        # Because of rounding shapes when using non-default stride or dilation,
        # convolution result must be truncated to convolution kernel size
        if self.conv_dims == 1:
            conv = conv.narrow(K_L_axis, 0, K_L)
        else:
            conv = conv.narrow(K_H_axis, 0, K_H).narrow(K_W_axis, 0, K_W)

        new_shape = [V, N, C_out, *spatial_dim_new]
        weight_grad = conv.reshape(*new_shape)

        if sum_batch:
            weight_grad = weight_grad.sum(1)

        return weight_grad

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        in_features = int(prod(module.input0.size()[1:]))
        out_features = int(prod(module.output.size()[1:]))

        mat = mat.reshape(out_features, *module.output.size()[1:])
        jac_t_mat = self.__jac_t(module, mat).reshape(out_features, in_features)

        mat_t_jac = jac_t_mat.t().reshape(in_features, *module.output.size()[1:])
        jac_t_mat_t_jac = self.__jac_t(module, mat_t_jac)
        jac_t_mat_t_jac = jac_t_mat_t_jac.reshape(in_features, in_features)

        return jac_t_mat_t_jac.t()
