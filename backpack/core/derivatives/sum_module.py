"""Contains derivatives for SumModule."""
from typing import List, Tuple

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
        subsampling: List[int] = None,
    ) -> Tensor:
        return mat

    def hessian_is_zero(self, module: SumModule) -> bool:  # noqa: D102
        """No gradients tracking parameters in SumModule, 2nd derivative thus =0"""
        return True
