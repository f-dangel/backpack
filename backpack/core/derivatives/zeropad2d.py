"""Partial derivatives for the ZeroPad2d function."""
from typing import List, Tuple

from einops import rearrange
from torch import Tensor
from torch.nn import ZeroPad2d, functional

from backpack.core.derivatives.basederivatives import BaseDerivatives


class ZeroPad2dDerivatives(BaseDerivatives):
    def hessian_is_zero(self, module):
        return True

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        _, C_out, H_out, W_out = module.output.shape
        _, in_c, in_x, in_y = module.input0.shape
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

    def _jac_t_mat_prod(
        self,
        module: ZeroPad2d,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        (W_top, W_bottom), (H_bottom, H_top) = self.__unpad_indices(module)
        return mat[:, :, :, W_top:W_bottom, H_bottom:H_top]

    def __unpad_indices(self, module):
        _, _, H_out, W_out = module.output.shape
        pad_left, pad_right, pad_top, pad_bottom = module.padding

        H_bottom, H_top = pad_left, W_out - pad_right
        W_top, W_bottom = pad_top, H_out - pad_bottom

        return (W_top, W_bottom), (H_bottom, H_top)

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        mat = rearrange(mat, "v n c h w -> (v n) c h w")
        pad_mat = functional.pad(mat, module.padding, "constant", module.value)
        return self.reshape_like_output(pad_mat, module)
