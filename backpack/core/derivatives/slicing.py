"""Contains derivatives of slicing operation."""
from typing import List, Tuple

from torch import Tensor, zeros

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.custom_module.slicing import Slicing
from backpack.utils.subsampling import subsample


class SlicingDerivatives(BaseDerivatives):
    """Derivatives of Slicing."""

    def _jac_t_mat_prod(
        self,
        module: Slicing,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        self.no_slice_batch_axis(module)

        input0 = module.input0
        result_shape = (mat.shape[0], *subsample(input0, subsampling=subsampling).shape)
        result = zeros(result_shape, device=input0.device, dtype=input0.dtype)
        result[(slice(None),) + module.slice_info] = mat

        return result

    @staticmethod
    def no_slice_batch_axis(module: Slicing):
        """Assert the batch axis is not sliced.

        Args:
            module: Slicing module.

        Raises:
            ValueError: If the batch axis is sliced.
        """
        slice_batch_axis = module.slice_info[0]

        if slice_batch_axis != slice(None):
            raise ValueError("Slicing the batch axis is not supported.")

    def hessian_is_zero(self, module: Slicing) -> bool:  # noqa: D102
        return True
