from ...utils import einsum
from torch.nn import Conv2d
from torch.nn.functional import conv_transpose2d
from .basejacobian import BaseJacobian


class Conv2dJacobian(BaseJacobian):

    def get_module(self):
        return Conv2d

    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        self.__check_sizes_input(mat, module)
        mat_as_conv = self.__reshape_for_conv(mat, module)
        jmp_as_conv = self.__apply_jacobian_of(module, mat_as_conv)
        self.__check_sizes_output(jmp_as_conv, module)

        return self.__reshape_for_matmul(jmp_as_conv, module)

    def __check_sizes_input(self, mat, module):
        batch, out_channels, out_x, out_y = module.output_shape
        assert tuple(mat.size())[:2] == (batch, out_channels * out_x * out_y)

    def __check_sizes_output(self, jmp, module):
        if tuple(jmp.size())[1:] != tuple(module.input0.size())[1:]:
            raise ValueError(
                "Size after conv_transpose does not match", "Got {}, and {}.",
                "Expected all dimensions to match, except for the first.".format(
                    jmp.size(), module.input0.size()))

    def __reshape_for_conv(self, bmat, module):
        batch, out_channels, out_x, out_y = module.output_shape
        num_classes = bmat.size(2)

        bmat = einsum('boc->cbo', (bmat, )).contiguous()
        bmat = bmat.view(num_classes * batch, out_channels, out_x, out_y)
        return bmat

    def __reshape_for_matmul(self, bconv, module):
        batch, _, _, _ = module.output_shape
        in_features = module.input0.numel() / batch
        bconv = bconv.view(-1, batch, in_features)
        bconv = einsum('cbi->bic', (bconv, ))
        return bconv

    def __apply_jacobian_of(self, module, mat):
        return conv_transpose2d(
            mat,
            module.weight.data,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups
        )
