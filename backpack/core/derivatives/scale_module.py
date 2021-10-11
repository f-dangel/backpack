"""Derivatives of ScaleModule (implies Identity)."""
from typing import List, Tuple, Union

from torch import Tensor
from torch.nn import Identity

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.custom_module.scale_module import ScaleModule


class ScaleModuleDerivatives(BaseDerivatives):
    """Derivatives of ScaleModule (implies Identity)."""

    def _jac_t_mat_prod(
        self,
        module: Union[ScaleModule, Identity],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        if isinstance(module, Identity):
            return mat
        else:
            return mat * module.weight
