import warnings

from torch.nn import ZeroPad2d
from torch.nn.functional import pad

from ...utils.utils import einsum, random_psd_matrix
from .basederivatives import BaseDerivatives
from .utils import jmp_unsqueeze_if_missing_dim


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

        result = result[:, idx_top:idx_bottom, idx_left:idx_right, :, idx_top:
                        idx_bottom, idx_left:idx_right].contiguous()

        return result.view(in_features, in_features)

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        # reshape feature dimension as output image
        batch, out_features, num_cols = mat.size()
        _, out_channels, out_x, out_y = module.output_shape
        assert out_features == out_channels * out_x * out_y
        mat = mat.view(batch, out_channels, out_x, out_y, num_cols)

        # remove padding by slicing
        pad_left, pad_right, pad_top, pad_bottom = module.padding
        idx_left, idx_right = pad_left, out_y - pad_right
        idx_top, idx_bottom = pad_top, out_x - pad_bottom
        mat_unpad = mat[:, :, idx_top:idx_bottom, idx_left:
                        idx_right, :].contiguous()

        # group in features
        _, in_channels, in_x, in_y = module.input0_shape
        in_features = in_channels * in_x * in_y
        return mat_unpad.view(batch, in_features, num_cols)

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        # group batch and column dimension of the matrix
        batch, in_features, num_cols = mat.size()
        mat = einsum('bic->bci', (mat)).contiguous()

        # reshape feature dimension as input image
        _, in_channels, in_x, in_y = module.input0_shape
        mat = mat.view(batch * num_cols, in_channels, in_x, in_y)

        # apply padding
        pad_mat = self.apply_padding(module, mat)

        # ungroup batch and column dimension
        _, out_channels, out_x, out_y = module.output_shape
        out_features = out_channels * out_x * out_y

        pad_mat = pad_mat.view(batch, num_cols, out_features)
        return einsum('bci->bic', (pad_mat)).contiguous()

    @staticmethod
    def apply_padding(module, input):
        return pad(input, module.padding, 'constant', module.value)
