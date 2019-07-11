import torch
from ...utils import einsum
from ...gradient.utils import conv as convUtils
from torch.nn import Conv2d
from torch.nn.functional import conv_transpose2d, conv2d
from .basederivatives import BaseDerivatives


class Conv2DDerivatives(BaseDerivatives):
    def get_module(self):
        return Conv2d

    def hessian_is_zero(self):
        return True

    # Jacobian-matrix product
    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        convUtils.check_sizes_input_jac(mat, module)
        mat_as_conv = self.__reshape_for_conv_in(mat, module)
        jmp_as_conv = self.__apply_jacobian_of(module, mat_as_conv)
        convUtils.check_sizes_output_jac(jmp_as_conv, module)

        return self.__reshape_for_matmul(jmp_as_conv, module)

    def __apply_jacobian_of(self, module, mat):
        return conv2d(
            mat,
            module.weight.data,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups)

    def __reshape_for_conv_in(self, bmat, module):
        batch, in_channels, in_x, in_y = module.input0.size()
        num_classes = bmat.size(2)
        bmat = einsum('boc->cbo', (bmat, )).contiguous()
        bmat = bmat.view(num_classes * batch, in_channels, in_x, in_y)
        return bmat

    def __reshape_for_matmul(self, bconv, module):
        batch = module.output_shape[0]
        out_features = torch.prod(module.output_shape) / batch
        bconv = bconv.view(-1, batch, out_features)
        bconv = einsum('cbi->bic', (bconv, ))
        return bconv

    # Transposed Jacobian-matrix product
    def jac_t_mat_prod(self, module, grad_input, grad_output, mat):
        convUtils.check_sizes_input_jac_t(mat, module)
        mat_as_conv = self.__reshape_for_conv_out(mat, module)
        jmp_as_conv = self.__apply_jacobian_t_of(module, mat_as_conv)
        convUtils.check_sizes_output_jac_t(jmp_as_conv, module)

        return self.__reshape_for_matmul_t(jmp_as_conv, module)

    def __reshape_for_conv_out(self, bmat, module):
        batch, out_channels, out_x, out_y = module.output_shape
        num_classes = bmat.size(2)

        bmat = einsum('boc->cbo', (bmat, )).contiguous()
        bmat = bmat.view(num_classes * batch, out_channels, out_x, out_y)
        return bmat

    def __reshape_for_matmul_t(self, bconv, module):
        batch = module.output_shape[0]
        in_features = module.input0.numel() / batch
        bconv = bconv.view(-1, batch, in_features)
        bconv = einsum('cbi->bic', (bconv, ))
        return bconv

    def __apply_jacobian_t_of(self, module, mat):
        return conv_transpose2d(
            mat,
            module.weight.data,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups)

    # TODO: Improve performance
    def bias_jac_mat_prod(self, module, grad_input, grad_output, mat):
        batch, out_channels, out_x, out_y = module.output_shape
        num_cols = mat.size(1)
        # mat has shape (out_channels, num_cols)
        # expand for each batch and for each channel
        jac_mat = mat.view(1, out_channels, 1, 1, num_cols)
        jac_mat = jac_mat.expand(batch, -1, out_x, out_y, -1).contiguous()
        return jac_mat.view(batch, -1, num_cols)

    # TODO: Improve performance
    def bias_jac_t_mat_prod(self, module, grad_input, grad_output, mat):
        batch, out_channels, out_x, out_y = module.output_shape
        num_cols = mat.size(2)
        shape = (batch, out_channels, out_x * out_y, num_cols)
        # mat has shape (batch, out_features, num_cols)
        # sum back over the pixels and batch dimensions
        jac_t_mat = mat.view(shape).sum([0, 2])
        return jac_t_mat

    # TODO: Improve performance, get rid of unfold
    def weight_jac_mat_prod(self, module, grad_input, grad_output, mat):
        batch, out_channels, out_x, out_y = module.output_shape
        out_features = out_channels * out_x * out_y
        num_cols = mat.size(1)
        jac_mat = mat.view(1, out_channels, -1, num_cols)
        jac_mat = jac_mat.expand(batch, out_channels, -1, -1)
        jac_mat = einsum('bij,bkic->bkjc', (convUtils.unfold_func(module)(
            module.input0), jac_mat)).contiguous()
        jac_mat = jac_mat.view(batch, out_features, num_cols)
        return jac_mat

    # TODO: Improve performance, get rid of unfold
    def weight_jac_t_mat_prod(self, module, grad_input, grad_output, mat):
        batch, out_channels, out_x, out_y = module.output_shape
        num_cols = mat.size(2)

        jac_t_mat = mat.view(batch, out_channels, -1, num_cols)
        jac_t_mat = einsum('bij,bkjc->kic', (convUtils.unfold_func(module)(
            module.input0), jac_t_mat)).contiguous()
        jac_t_mat = jac_t_mat.view(module.weight.numel(), num_cols)
        return jac_t_mat
