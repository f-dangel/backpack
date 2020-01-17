import numpy
from torch.nn import ZeroPad2d
from torch.nn.functional import pad

from backpack.core.derivatives.utils import (
    jac_t_mat_prod_accept_vectors,
    jac_mat_prod_accept_vectors,
)

from backpack.utils.einsum import einsum
from backpack.core.derivatives.basederivatives import BaseDerivatives


class ZeroPad2dDerivatives(BaseDerivatives):
    def get_module(self):
        return ZeroPad2d

    def hessian_is_zero(self):
        return True

    # TODO: Require tests
    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        _, out_c, out_x, out_y = module.output_shape
        _, in_c, in_x, in_y = module.input0_shape
        in_features = in_c * in_x * in_y

        # slicing indices
        pad_left, pad_right, pad_top, pad_bottom = module.padding
        idx_left, idx_right = pad_left, out_y - pad_right
        idx_top, idx_bottom = pad_top, out_x - pad_bottom

        result = mat.view(out_c, out_x, out_y, out_c, out_x, out_y)

        result = result[
            :,
            idx_top:idx_bottom,
            idx_left:idx_right,
            :,
            idx_top:idx_bottom,
            idx_left:idx_right,
        ].contiguous()

        return result.view(in_features, in_features)

    @jac_t_mat_prod_accept_vectors
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        new_convention = True

        # reshape feature dimension as output image
        if new_convention:
            num_cols = mat.size(0)
            batch = mat.size(1)
            out_features = numpy.prod(mat.shape[2:])
            _, out_channels, out_x, out_y = module.output_shape
            assert out_features == out_channels * out_x * out_y
            shape = (num_cols, batch, out_channels, out_x, out_y)
        else:
            batch, out_features, num_cols = mat.size()
            _, out_channels, out_x, out_y = module.output_shape
            assert out_features == out_channels * out_x * out_y
            shape = (batch, out_channels, out_x, out_y, num_cols)

        mat = mat.view(shape)

        # remove padding by slicing
        pad_left, pad_right, pad_top, pad_bottom = module.padding
        idx_left, idx_right = pad_left, out_y - pad_right
        idx_top, idx_bottom = pad_top, out_x - pad_bottom

        if new_convention:
            mat_unpad = mat[
                :, :, :, idx_top:idx_bottom, idx_left:idx_right
            ].contiguous()
        else:
            mat_unpad = mat[
                :, :, idx_top:idx_bottom, idx_left:idx_right, :
            ].contiguous()

        if new_convention:
            return mat_unpad
        else:
            # group in features
            _, in_channels, in_x, in_y = module.input0_shape
            in_features = in_channels * in_x * in_y
            return mat_unpad.view(batch, in_features, num_cols)

    @jac_mat_prod_accept_vectors
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        new_convention = True

        # group batch and column dimension of the matrix
        if new_convention:
            batch = mat.shape[1]
            num_cols = mat.shape[0]
        else:
            batch, in_features, num_cols = mat.size()
            mat = einsum("bic->bci", (mat))

        # reshape feature dimension as input image
        _, in_channels, in_x, in_y = module.input0_shape
        mat = mat.contiguous().view(batch * num_cols, in_channels, in_x, in_y)

        # apply padding
        pad_mat = self.apply_padding(module, mat)

        # ungroup batch and column dimension

        _, out_channels, out_x, out_y = module.output_shape
        out_features = out_channels * out_x * out_y

        if new_convention:
            shape = (num_cols,) + tuple(module.output_shape)
            return pad_mat.view(shape)
        else:
            pad_mat = pad_mat.view(batch, num_cols, out_features)
            return einsum("bci->bic", (pad_mat)).contiguous()

    @staticmethod
    def apply_padding(module, input):
        return pad(input, module.padding, "constant", module.value)
