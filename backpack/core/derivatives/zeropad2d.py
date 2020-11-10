from torch.nn import functional

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.utils.ein import eingroup


class ZeroPad2dDerivatives(BaseDerivatives):
    def hessian_is_zero(self):
        return True

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        _, C_out, H_out, W_out = module.output_shape
        _, in_c, in_x, in_y = module.input0_shape
        in_features = in_c * in_x * in_y

        result = mat.view(C_out, H_out, W_out, C_out, H_out, W_out)

        (W_top, W_bottom), (H_bottom, H_top) = self.__unpad_indices(module)
        result = result[
            :,
            W_top:W_bottom,
            H_bottom:H_top,
            :,
            W_top:W_bottom,
            H_bottom:H_top,
        ].contiguous()

        return result.view(in_features, in_features)

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        (W_top, W_bottom), (H_bottom, H_top) = self.__unpad_indices(module)
        return mat[:, :, :, W_top:W_bottom, H_bottom:H_top]

    def __unpad_indices(self, module):
        _, _, H_out, W_out = module.output_shape
        pad_left, pad_right, pad_top, pad_bottom = module.padding

        H_bottom, H_top = pad_left, W_out - pad_right
        W_top, W_bottom = pad_top, H_out - pad_bottom

        return (W_top, W_bottom), (H_bottom, H_top)

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        mat = eingroup("v,n,c,h,w->vn,c,h,w", mat)
        pad_mat = functional.pad(mat, module.padding, "constant", module.value)
        return self.reshape_like_output(pad_mat, module)
