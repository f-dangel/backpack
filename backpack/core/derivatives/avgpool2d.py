"""The code relies on the insight that average pooling can be understood as
convolution over single channels with a constant kernel."""

import warnings

import torch.nn
from torch.nn import AvgPool2d, Conv2d, ConvTranspose2d

from ...utils import conv as convUtils
from ...utils.utils import einsum, random_psd_matrix
from .basederivatives import BaseDerivatives
from .utils import jmp_unsqueeze_if_missing_dim


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
        result = einsum('cxyf->fcxy', (result, )).contiguous()
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

    # Jacobian-matrix product
    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        assert module.count_include_pad, "Might now work for exotic hyperparameters of AvgPool2d, like count_include_pad=False"

        convUtils.check_sizes_input_jac(mat, module)
        mat_as_pool = self.__reshape_for_conv(mat, module)
        jmp_as_pool = self.__apply_jacobian_of(module, mat_as_pool)

        batch, channels, out_x, out_y = module.output_shape
        num_classes = mat.size(2)
        assert jmp_as_pool.size(0) == num_classes * batch * channels
        assert jmp_as_pool.size(1) == 1
        assert jmp_as_pool.size(2) == out_x
        assert jmp_as_pool.size(3) == out_y

        return self.__reshape_for_matmul(jmp_as_pool, module)

    def __reshape_for_conv(self, mat, module):
        """Create fake single-channel images, grouping batch,
        class and channel dimension."""
        batch, in_channels, in_x, in_y = module.input0.size()
        num_columns = mat.size(-1)

        # 'fake' image for convolution
        # (batch * class * channel, 1,  out_x, out_y)
        return einsum('bic->bci',
                      mat).contiguous().view(batch * num_columns * in_channels,
                                             1, in_x, in_y)

    def __reshape_for_matmul(self, mat, module):
        """Ungroup dimensions after application of Jacobian."""
        batch, channels, out_x, out_y = module.output_shape
        features = channels * out_x * out_y
        # mat is of shape (batch * class * channel, 1, out_x, out_y)
        # move class dimension to last
        mat_view = mat.view(batch, -1, features)
        return einsum('bci->bic', mat_view).contiguous()

    def __apply_jacobian_of(self, module, mat):
        conv2d = Conv2d(in_channels=1,
                        out_channels=1,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        bias=False).to(module.input0.device)

        conv2d.weight.requires_grad = False
        avg_kernel = torch.ones_like(conv2d.weight) / conv2d.weight.numel()
        conv2d.weight.data = avg_kernel

        return conv2d(mat)

    # Transpose Jacobian-matrix product
    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):

        assert module.count_include_pad, "Might now work for exotic hyperparameters of AvgPool2d, like count_include_pad=False"

        convUtils.check_sizes_input_jac_t(mat, module)
        mat_as_pool = self.__reshape_for_conv_t(mat, module)
        jmp_as_pool = self.__apply_jacobian_t_of(module, mat_as_pool)

        batch, channels, in_x, in_y = module.input0.size()
        num_classes = mat.size(2)
        assert jmp_as_pool.size(0) == num_classes * batch * channels
        assert jmp_as_pool.size(1) == 1
        assert jmp_as_pool.size(2) == in_x
        assert jmp_as_pool.size(3) == in_y

        return self.__reshape_for_matmul_t(jmp_as_pool, module)

    def __reshape_for_conv_t(self, mat, module):
        """Create fake single-channel images, grouping batch,
        class and channel dimension."""
        batch, out_channels, out_x, out_y = module.output_shape
        num_classes = mat.size(-1)

        # 'fake' image for convolution
        # (batch * class * channel, 1,  out_x, out_y)
        return einsum('bic->bci', mat).contiguous().view(
            batch * num_classes * out_channels, 1, out_x, out_y)

    def __reshape_for_matmul_t(self, mat, module):
        """Ungroup dimensions after application of Jacobian."""
        batch, channels, in_x, in_y = module.input0.size()
        features = channels * in_x * in_y
        # mat is of shape (batch * class * channel, 1,  in_x, in_y)
        # move class dimension to last
        mat_view = mat.view(batch, -1, features)
        return einsum('bci->bic', mat_view).contiguous()

    def __apply_jacobian_t_of(self, module, mat):
        _, _, in_x, in_y = module.input0.size()
        output_size = (mat.size(0), 1, in_x, in_y)

        conv2d_t = ConvTranspose2d(in_channels=1,
                                   out_channels=1,
                                   kernel_size=module.kernel_size,
                                   stride=module.stride,
                                   padding=module.padding,
                                   bias=False).to(module.input0.device)

        conv2d_t.weight.requires_grad = False
        avg_kernel = torch.ones_like(conv2d_t.weight) / conv2d_t.weight.numel()
        conv2d_t.weight.data = avg_kernel

        return conv2d_t(mat, output_size=output_size)
