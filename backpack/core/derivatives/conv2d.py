from torch.nn import Conv2d
from torch.nn.functional import conv2d, conv_transpose2d

from backpack.core.derivatives.utils import (
    weight_jac_t_mat_prod_accept_vectors,
    weight_jac_mat_prod_accept_vectors,
    bias_jac_t_mat_prod_accept_vectors,
    bias_jac_mat_prod_accept_vectors,
    jac_t_mat_prod_accept_vectors,
    jac_mat_prod_accept_vectors,
)

from backpack.utils import conv as convUtils
from backpack.utils.einsum import einsum, eingroup
from backpack.core.derivatives.basederivatives import BaseDerivatives


class Conv2DDerivatives(BaseDerivatives):
    def get_module(self):
        return Conv2d

    def hessian_is_zero(self):
        return True

    def get_unfolded_input(self, module):
        return convUtils.unfold_func(module)(module.input0)

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

    @jac_mat_prod_accept_vectors
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = eingroup("v,n,c,h,w->vn,c,h,w", mat)
        jmp_as_conv = conv2d(
            mat_as_conv,
            module.weight.data,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        return self.view_like_output(jmp_as_conv, module)

    @jac_t_mat_prod_accept_vectors
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = eingroup("v,n,c,h,w->vn,c,h,w", mat)
        jmp_as_conv = conv_transpose2d(
            mat_as_conv,
            module.weight.data,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        return self.view_like_input(jmp_as_conv, module)

    # TODO: Improve performance
    @bias_jac_mat_prod_accept_vectors
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

    @bias_jac_t_mat_prod_accept_vectors
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
    @weight_jac_mat_prod_accept_vectors
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

    @weight_jac_t_mat_prod_accept_vectors
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
