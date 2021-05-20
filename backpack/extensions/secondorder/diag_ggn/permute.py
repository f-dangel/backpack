"""Module defining DiagGGNPermute."""
from typing import Tuple

from torch import Tensor
from torch.nn import RNN

from backpack.core.derivatives.permute import PermuteDerivatives
from backpack.extensions.module_extension import ModuleExtension
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNPermute(DiagGGNBaseModule):
    """DiagGGN extension of Permute."""

    def __init__(self):
        """Initialize."""
        super().__init__(derivatives=PermuteDerivatives())

    def backpropagate(
        self,
        ext: ModuleExtension,
        module: RNN,
        grad_inp: Tuple[Tensor],
        grad_out: Tuple[Tensor],
        backproped: Tensor,
    ) -> Tensor:
        """Propagates secondorder information back.

        Args:
            ext: extension
            module: module through which to backpropagate
            grad_inp: input gradients
            grad_out: output gradients
            backproped: backpropagated information

        Returns:
            derivative wrt input
        """
        return super().backpropagate(ext, module, grad_inp, grad_out, backproped)
