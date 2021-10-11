"""Module containing derivatives of Permute."""
from typing import List, Tuple

from torch import Tensor, argsort

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.custom_module.permute import Permute


class PermuteDerivatives(BaseDerivatives):
    """Derivatives of Permute."""

    def _jac_t_mat_prod(
        self,
        module: Permute,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        return mat.permute(
            [0] + [element + 1 for element in argsort(Tensor(module.dims))]
        )

    def _jac_mat_prod(
        self, module: Permute, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        return mat.permute([0] + [element + 1 for element in module.dims])
