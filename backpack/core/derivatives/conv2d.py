import torch
from torch.nn import Conv2d
from torch.nn.functional import conv2d, conv_transpose2d

from backpack.core.derivatives.utils import (
    bias_jac_t_new_shape_convention,
    bias_jac_new_shape_convention,
    jac_new_shape_convention,
    jac_t_new_shape_convention,
    weight_jac_t_new_shape_convention,
    weight_jac_new_shape_convention,
)
from backpack.utils.unsqueeze import jmp_unsqueeze_if_missing_dim

from ...utils import conv as convUtils
from ...utils.einsum import einsum
from .basederivatives import BaseDerivatives


class Conv2DDerivatives(BaseDerivatives):
    def get_module(self):
        return Conv2d

    def hessian_is_zero(self):
        return True

    def get_weight_data(self, module):
        return module.weight.data

    def get_input(self, module):
        return module.input0

    def get_unfolded_input(self, module):
        return convUtils.unfold_func(module)(self.get_input(module))

    # TODO: Require tests
    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        _, in_c, in_x, in_y = module.input0.size()
        in_features = in_c * in_x * in_y
        _, out_c, out_x, out_y = module.output.size()
        out_features = out_c * out_x * out_y

        # 1) apply conv_transpose to multiply with W^T
        result = mat.view(out_c, out_x, out_y, out_features)
        result = einsum("cxyf->fcxy", (result,))
        # result: W^T mat
        result = self.__apply_jacobian_t_of(module, result).view(
            out_features, in_features
        )

        # 2) transpose: mat^T W
        result = result.t()

        # 3) apply conv_transpose
        result = result.view(in_features, out_c, out_x, out_y)
        result = self.__apply_jacobian_t_of(module, result)

        # 4) transpose to obtain W^T mat W
        return result.view(in_features, in_features).t()

    # Jacobian-matrix product
    @jmp_unsqueeze_if_missing_dim(mat_dim=2)
    @jac_new_shape_convention
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        new_convention = True

        convUtils.check_sizes_input_jac(mat, module, new_convention=new_convention)
        mat_as_conv = self.__reshape_for_conv_in(
            mat, module, new_convention=new_convention
        )
        jmp_as_conv = self.__apply_jacobian_of(module, mat_as_conv)
        convUtils.check_sizes_output_jac(jmp_as_conv, module)

        return self.__reshape_for_matmul(
            jmp_as_conv, module, new_convention=new_convention
        )

    def __apply_jacobian_of(self, module, mat):
        return conv2d(
            mat,
            self.get_weight_data(module),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    def __reshape_for_conv_in(self, bmat, module, new_convention=False):
        batch, in_channels, in_x, in_y = module.input0.size()

        if new_convention:
            num_classes = bmat.size(0)
        else:
            num_classes = bmat.size(2)
            bmat = einsum("boc->cbo", (bmat,))

        bmat = bmat.contiguous().view(num_classes * batch, in_channels, in_x, in_y)
        return bmat

    def __reshape_for_matmul(self, bconv, module, new_convention=False):
        if new_convention:
            shape = (-1,) + tuple(module.output_shape)
            bconv = bconv.view(shape)
            pass
        else:
            batch = module.output_shape[0]
            out_features = torch.prod(module.output_shape) / batch
            bconv = bconv.view(-1, batch, out_features)
            bconv = einsum("cbi->bic", (bconv,))
        return bconv

    # Transposed Jacobian-matrix product
    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    @jac_t_new_shape_convention
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        new_convention = True

        convUtils.check_sizes_input_jac_t(mat, module, new_convention=new_convention)
        mat_as_conv = self.__reshape_for_conv_out(
            mat, module, new_convention=new_convention
        )
        jmp_as_conv = self.__apply_jacobian_t_of(module, mat_as_conv)
        convUtils.check_sizes_output_jac_t(jmp_as_conv, module)

        return self.__reshape_for_matmul_t(
            jmp_as_conv, module, new_convention=new_convention
        )

    def __reshape_for_conv_out(self, bmat, module, new_convention=False):
        batch, out_channels, out_x, out_y = module.output_shape
        if new_convention:
            num_classes = bmat.size(0)
        else:
            num_classes = bmat.size(2)
            bmat = einsum("boc->cbo", (bmat,)).contiguous()
        bmat = bmat.contiguous().view(num_classes * batch, out_channels, out_x, out_y)
        return bmat

    def __reshape_for_matmul_t(self, bconv, module, new_convention=False):
        batch = module.output_shape[0]
        if new_convention:
            shape = (-1, batch) + tuple(module.input0_shape[1:])
            return bconv.view(shape)
        else:
            in_features = module.input0.numel() / batch
            bconv = bconv.view(-1, batch, in_features)
            bconv = einsum("cbi->bic", (bconv,))
            return bconv

    def __apply_jacobian_t_of(self, module, mat):
        return conv_transpose2d(
            mat,
            self.get_weight_data(module),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    # TODO: Improve performance
    @jmp_unsqueeze_if_missing_dim(mat_dim=2)
    @bias_jac_new_shape_convention
    def bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        new_convention = True

        batch, out_channels, out_x, out_y = module.output_shape
        if new_convention:
            # mat has shape (V, out_channels)
            # expand for each batch and for each channel
            jac_mat = mat.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            return jac_mat.expand(-1, batch, -1, out_x, out_y).contiguous()
        else:
            num_cols = mat.size(1)
            # mat has shape (out_channels, num_cols)
            # expand for each batch and for each channel
            jac_mat = mat.view(1, out_channels, 1, 1, num_cols)
            jac_mat = jac_mat.expand(batch, -1, out_x, out_y, -1).contiguous()
            return jac_mat.view(batch, -1, num_cols)

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    @bias_jac_t_new_shape_convention
    def bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        new_convention = True

        if new_convention:
            if sum_batch:
                sum_dims = [1, 3, 4]
            else:
                sum_dims = [3, 4]
            return mat.sum(sum_dims)
        else:
            batch, out_channels, out_x, out_y = module.output_shape
            num_cols = mat.size(2)
            shape = (batch, out_channels, out_x * out_y, num_cols)
            # mat has shape (batch, out_features, num_cols)
            # sum back over the pixels and batch dimensions
            sum_dims = [0, 2] if sum_batch is True else [2]
            return mat.view(shape).sum(sum_dims)

    # TODO: Improve performance, get rid of unfold, use conv
    @jmp_unsqueeze_if_missing_dim(mat_dim=2)
    @weight_jac_new_shape_convention
    def weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        new_convention = True

        batch, out_channels, out_x, out_y = module.output_shape
        if new_convention:
            num_cols = mat.size(0)
            jac_mat = mat.view(num_cols, out_channels, -1)
            X = self.get_unfolded_input(module)

            jac_mat = einsum("bij,cki->cbkj", (X, jac_mat)).contiguous()
            jac_mat = jac_mat.view(num_cols, batch, out_channels, out_x, out_y)
        else:
            out_features = out_channels * out_x * out_y
            num_cols = mat.size(1)
            jac_mat = mat.view(1, out_channels, -1, num_cols)
            jac_mat = jac_mat.expand(batch, out_channels, -1, -1)

            X = self.get_unfolded_input(module)
            jac_mat = einsum("bij,bkic->bkjc", (X, jac_mat)).contiguous()
            jac_mat = jac_mat.view(batch, out_features, num_cols)
        return jac_mat

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    @weight_jac_t_new_shape_convention
    def weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Unintuitive, but faster due to conv operation."""
        new_convention = True

        batch, out_channels, out_x, out_y = module.output_shape
        _, in_channels, in_x, in_y = module.input0.shape
        k_x, k_y = module.kernel_size

        if new_convention:
            num_cols = mat.shape[0]
        else:
            num_cols = mat.shape[-1]
            shape = (batch, out_channels, out_x, out_y, num_cols)
            mat = mat.view(shape)

        if new_convention:
            pass
        else:
            mat = einsum("boxyc->cboxy", (mat,))

        mat = mat.contiguous().view(num_cols * batch, out_channels, out_x, out_y)

        mat = mat.repeat(1, in_channels, 1, 1)
        mat = mat.view(num_cols * batch * out_channels * in_channels, 1, out_x, out_y)

        input = module.input0.view(1, -1, in_x, in_y).repeat(1, num_cols, 1, 1)

        grad_weight = conv2d(
            input,
            mat,
            None,
            module.dilation,
            module.padding,
            module.stride,
            in_channels * batch * num_cols,
        )

        if new_convention:
            grad_weight = grad_weight.view(
                num_cols, batch, out_channels, in_channels, k_x, k_y
            )
            if sum_batch is True:
                grad_weight = grad_weight.sum(1)
                batch = 1

            grad_weight = grad_weight.view(
                num_cols, batch, in_channels, out_channels, k_x, k_y
            )
            grad_weight = einsum("cbmnxy->cbnmxy", grad_weight).contiguous()

            if sum_batch is True:
                grad_weight = grad_weight.squeeze(1)

            return grad_weight

        else:
            grad_weight = grad_weight.view(
                num_cols, batch, out_channels * in_channels, k_x, k_y
            )
            if sum_batch is True:
                grad_weight = grad_weight.sum(1)
                batch = 1

            grad_weight = grad_weight.view(
                num_cols, batch, in_channels, out_channels, k_x, k_y
            )
            grad_weight = einsum("cbmnxy->bnmxyc", grad_weight).contiguous()

            grad_weight = grad_weight.view(
                batch, in_channels * out_channels * k_x * k_y, num_cols
            )

            if sum_batch is True:
                grad_weight = grad_weight.squeeze(0)

            return grad_weight
