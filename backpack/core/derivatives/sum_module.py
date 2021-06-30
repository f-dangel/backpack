"""Contains derivatives for SumModule."""
from typing import Tuple

from torch import Tensor

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.custom_module.branching import SumModule


class SumModuleDerivatives(BaseDerivatives):
    """Contains derivatives for SumModule."""

    def _jac_t_mat_prod(
        self,
        module: SumModule,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
    ) -> Tensor:
        return mat
