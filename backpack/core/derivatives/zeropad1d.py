from torch.nn import functional

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.utils.ein import eingroup


class ZeroPad1dDerivatives(BaseDerivatives):
    def hessian_is_zero(self):
        return True

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        (D_top, D_bottom) = self.__unpad_indices(module)
        return mat[:, :, :, D_top:D_bottom]

    def __unpad_indices(self, module):
        _, _, D_out = module.output.shape
        pad_left, pad_right = module.padding

        D_bottom, D_top = pad_left, D_out - pad_right

        return (D_top, D_bottom)

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        mat = eingroup("v,n,c,d->vn,c,d", mat)
        pad_mat = functional.pad(mat, module.padding, "constant", module.value)
        return self.reshape_like_output(pad_mat, module)
