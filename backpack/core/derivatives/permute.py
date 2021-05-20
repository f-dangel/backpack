"""Module containing derivatives of Permute."""
from copy import deepcopy

from torch import Tensor, argsort

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.custom_module.permute import Permute


class PermuteDerivatives(BaseDerivatives):
    """Derivatives of Permute."""

    def _jac_t_mat_prod(self, module: Permute, g_inp, g_out, mat):
        return deepcopy(
            mat.permute([0] + [element + 1 for element in argsort(Tensor(module.dims))])
        )

    def _jac_mat_prod(self, module: Permute, g_inp, g_out, mat):
        return deepcopy(
            mat.permute([0] + [element + 1 for element in module.dims]).detach()
        )
