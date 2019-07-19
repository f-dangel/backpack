import torch
from ...core.layers import Conv2dConcat
from ...utils.utils import einsum
from ...utils import conv as convUtils
from torch.nn import Conv2d
from torch.nn.functional import conv_transpose2d, conv2d
from .basederivatives import BaseDerivatives

from .utils import unsqueeze_if_missing_dim


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
            self.get_weight_data(module),
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
            self.get_weight_data(module),
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

    @unsqueeze_if_missing_dim(mat_dim=3)
    def bias_jac_t_mat_prod(self,
                            module,
                            grad_input,
                            grad_output,
                            mat,
                            sum_batch=True):
        batch, out_channels, out_x, out_y = module.output_shape
        num_cols = mat.size(2)
        shape = (batch, out_channels, out_x * out_y, num_cols)
        # mat has shape (batch, out_features, num_cols)
        # sum back over the pixels and batch dimensions
        sum_dims = [0, 2] if sum_batch is True else [2]
        return mat.view(shape).sum(sum_dims)

    # TODO: Improve performance, get rid of unfold
    def weight_jac_mat_prod(self, module, grad_input, grad_output, mat):
        batch, out_channels, out_x, out_y = module.output_shape
        out_features = out_channels * out_x * out_y
        num_cols = mat.size(1)
        jac_mat = mat.view(1, out_channels, -1, num_cols)
        jac_mat = jac_mat.expand(batch, out_channels, -1, -1)

        X = self.get_unfolded_input(module)
        jac_mat = einsum('bij,bkic->bkjc', (X, jac_mat)).contiguous()
        jac_mat = jac_mat.view(batch, out_features, num_cols)
        return jac_mat

    @unsqueeze_if_missing_dim(mat_dim=3)
    def weight_jac_t_mat_prod2(self,
                               module,
                               grad_input,
                               grad_output,
                               mat,
                               sum_batch=True):
        """Intuitive, using unfold operation."""
        batch, out_channels, out_x, out_y = module.output_shape
        _, in_channels, in_x, in_y = module.input0.shape
        num_cols = mat.shape[-1]

        jac_t_mat = mat.view(batch, out_channels, -1, num_cols)

        equation = 'bij,bkjc->kic' if sum_batch is True else 'bij,bkjc->bkic'

        X = self.get_unfolded_input(module)
        jac_t_mat = einsum(equation, (X, jac_t_mat)).contiguous()

        sum_shape = [module.weight.numel(), num_cols]
        shape = sum_shape if sum_batch is True else [batch] + sum_shape

        jac_t_mat = jac_t_mat.view(shape)
        return jac_t_mat

    @unsqueeze_if_missing_dim(mat_dim=3)
    def weight_jac_t_mat_prod(self,
                              module,
                              grad_input,
                              grad_output,
                              mat,
                              sum_batch=True):
        """Unintuitive, but faster due to conv operation."""
        batch, out_channels, out_x, out_y = module.output_shape
        _, in_channels, in_x, in_y = module.input0.shape
        k_x, k_y = module.kernel_size
        num_cols = mat.shape[-1]

        mat = mat.view(batch, out_channels, out_x, out_y, num_cols)
        mat = einsum('boxyc->cboxy', (mat, )).contiguous().view(
            num_cols * batch, out_channels, out_x, out_y)

        mat = mat.repeat(1, in_channels, 1, 1)
        mat = mat.view(num_cols * batch * out_channels * in_channels, 1, out_x,
                       out_y)

        input = module.input0.view(1, -1, in_x, in_y).repeat(1, num_cols, 1, 1)

        grad_weight = conv2d(input, mat, None, module.dilation, module.padding,
                             module.stride, in_channels * batch * num_cols)

        grad_weight = grad_weight.view(num_cols, batch,
                                       out_channels * in_channels, k_x, k_y)
        if sum_batch is True:
            grad_weight = grad_weight.sum(1)
            batch = 1

        grad_weight = grad_weight.view(num_cols, batch, in_channels,
                                       out_channels, k_x, k_y)
        grad_weight = einsum('cbmnxy->bnmxyc', grad_weight).contiguous()

        grad_weight = grad_weight.view(
            batch, in_channels * out_channels * k_x * k_y, num_cols)

        if sum_batch is True:
            grad_weight = grad_weight.squeeze(0)

        return grad_weight


class Conv2DConcatDerivatives(Conv2DDerivatives):
    # override
    def get_module(self):
        return Conv2dConcat

    # override
    def get_unfolded_input(self, module):
        """Return homogeneous input, if bias exists """
        X = convUtils.unfold_func(module)(self.get_input(module))
        if module.has_bias():
            return module.append_ones(X)
        else:
            return X

    # override
    def get_weight_data(self, module):
        return module._slice_weight().data

    def bias_jac_mat_prod(self, module, grad_input, grad_output, mat):
        raise RuntimeError("Bias is concatenated with weight matrix")

    def bias_jac_t_mat_prod(self,
                            module,
                            grad_input,
                            grad_output,
                            mat,
                            sum_batch=True):
        raise RuntimeError("Bias is concatenated with weight matrix")

    # override
    @unsqueeze_if_missing_dim(mat_dim=3)
    def weight_jac_t_mat_prod(self,
                              module,
                              grad_input,
                              grad_output,
                              mat,
                              sum_batch=True):
        # TODO: Figure out how to use weight_jac_t_mat_prod to treat
        # concatenated parameter of weight/bias
        return self.weight_jac_t_mat_prod2(
            module, grad_input, grad_output, mat, sum_batch=sum_batch)
