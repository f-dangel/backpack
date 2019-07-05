from ...utils import einsum
from ..utils import conv as convUtils
from torch.nn import Conv2d
from torch.nn.functional import conv_transpose2d
from .basejacobian import BaseJacobian


class Conv2dJacobian(BaseJacobian):

    def get_module(self):
        return Conv2d

    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        convUtils.check_sizes_input(mat, module)
        mat_as_conv = self.__reshape_for_conv(mat, module)
        jmp_as_conv = self.__apply_jacobian_of(module, mat_as_conv)
        convUtils.check_sizes_output(jmp_as_conv, module)

        return self.__reshape_for_matmul(jmp_as_conv, module)

    def __reshape_for_conv(self, bmat, module):
        batch, out_channels, out_x, out_y = module.output_shape
        num_classes = bmat.size(2)

        bmat = einsum('boc->cbo', (bmat, )).contiguous()
        bmat = bmat.view(num_classes * batch, out_channels, out_x, out_y)
        return bmat

    def __reshape_for_matmul(self, bconv, module):
        batch = module.output_shape[0]
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

    def hessian_is_zero(self):
        return True
