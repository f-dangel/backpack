"""Derivatives of the identity module."""
from typing import Tuple, Union

from torch import Tensor
from torch.nn import Identity

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.custom_module.scale_module import ScaleModule


class ScaleModuleDerivatives(BaseDerivatives):
    def _jac_t_mat_prod(
        self,
        module: Union[ScaleModule, Identity],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
    ) -> Tensor:
        if isinstance(module, Identity):
            return mat
        else:
            return mat * module.weight
