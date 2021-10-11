"""Partial derivatives of the flatten layer."""
from typing import List, Tuple

from torch import Tensor
from torch.nn import Flatten

from backpack.core.derivatives.basederivatives import BaseDerivatives


class FlattenDerivatives(BaseDerivatives):
    def hessian_is_zero(self, module):
        return True

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        return mat

    def _jac_t_mat_prod(
        self,
        module: Flatten,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        return self.reshape_like_input(mat, module, subsampling=subsampling)

    def _jac_mat_prod(
        self,
        module: Flatten,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
    ) -> Tensor:
        return self.reshape_like_output(mat, module)
