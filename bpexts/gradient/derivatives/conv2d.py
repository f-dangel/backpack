import torch
from ...utils import einsum
from ..utils import conv as convUtils
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
