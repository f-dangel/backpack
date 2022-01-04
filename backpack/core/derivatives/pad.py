"""Contains derivatives of N-dimensional padding."""

from typing import List, Sequence, Tuple

from torch import Tensor

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.custom_module.pad import Pad


class PadDerivatives(BaseDerivatives):
    """Derivatives of Pad."""

    def _jac_t_mat_prod(
        self,
        module: Pad,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        self.no_pad_batch_axis(module)

        return self.unpad(mat, module.pad, module.mode, module.value)

    @staticmethod
    def no_pad_batch_axis(module: Pad):
        """Assert the batch axis is not padded.

        Args:
            module: Pad module.

        Raises:
            ValueError: If the batch axis is padded.
        """
        num_pad_axes = len(module.pad) // 2
        if num_pad_axes == module.input0.dim():
            raise ValueError("Padding the batch axis is not supported.")

    @staticmethod
    def unpad(tensor: Tensor, pad: Sequence[int], mode: str, value: float) -> Tensor:
        """Remove padding from a tensor.

        Undoes the operation ``torch.nn.functional.pad``.

        Args:
            pad: Tuple of even length specifying the padding.
            mode: Padding mode.
            value: Fill value for constant padding.

        Returns:
            Unpadded tensor.

        Raises:
            NotImplementedError: If padding mode is not constant.
        """
        if mode != "constant":
            raise NotImplementedError("Only mode='constant' is supported.")

        pad_axes = len(pad) // 2
        unaffected = tensor.dim() - pad_axes

        no_slice = [slice(None) for _ in range(unaffected)]
        unpad_slice = []

        for affected in range(pad_axes):
            pad_start, pad_end = pad[2 * affected : 2 * affected + 2]
            dim = tensor.shape[tensor.dim() - 1 - affected]
            unpad_slice.insert(0, slice(pad_start, dim - pad_end))

        return tensor[no_slice + unpad_slice]

    def hessian_is_zero(self, module: Pad) -> bool:  # noqa: D102
        return True
