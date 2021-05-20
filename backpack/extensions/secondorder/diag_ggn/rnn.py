"""Module implementing GGN."""
from typing import Tuple

from torch import Tensor
from torch.nn import RNN

from backpack.core.derivatives.rnn import RNNDerivatives
from backpack.extensions.module_extension import ModuleExtension
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNRNN(DiagGGNBaseModule):
    """Calculating GGN derivative."""

    def __init__(self):
        """Initialize."""
        super().__init__(
            derivatives=RNNDerivatives(),
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
        )

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

    def bias_ih_l0(
        self,
        ext: ModuleExtension,
        module: RNN,
        grad_inp: Tuple[Tensor],
        grad_out: Tuple[Tensor],
        backproped: Tensor,
    ) -> Tensor:
        """Returns diagonal of GGN.

        Args:
            ext: extension
            module: module through which to backpropagate
            grad_inp: input gradients
            grad_out: output gradients
            backproped: backpropagated information

        Returns:
            diagonal
        """
        JS = self.derivatives.bias_ih_l0_jac_t_mat_prod(
            module, grad_inp, grad_out, backproped, sum_batch=False
        )
        return (JS ** 2).sum(axis=0).sum(axis=0)

    def bias_hh_l0(
        self,
        ext,
        module: RNN,
        grad_inp: Tuple[Tensor],
        grad_out: Tuple[Tensor],
        backproped: Tensor,
    ) -> Tensor:
        """Returns diagonal of GGN.

        Args:
            ext: extension
            module: module through which to backpropagate
            grad_inp: input gradients
            grad_out: output gradients
            backproped: backpropagated information

        Returns:
            diagonal
        """
        JS = self.derivatives.bias_hh_l0_jac_t_mat_prod(
            module, grad_inp, grad_out, backproped, sum_batch=False
        )
        return (JS ** 2).sum(axis=0).sum(axis=0)

    def weight_ih_l0(
        self,
        ext,
        module: RNN,
        grad_inp: Tuple[Tensor],
        grad_out: Tuple[Tensor],
        backproped: Tensor,
    ) -> Tensor:
        """Returns diagonal of GGN.

        Args:
            ext: extension
            module: module through which to backpropagate
            grad_inp: input gradients
            grad_out: output gradients
            backproped: backpropagated information

        Returns:
            diagonal
        """
        JS = self.derivatives.weight_ih_l0_jac_t_mat_prod(
            module, grad_inp, grad_out, backproped, sum_batch=False
        )
        return (JS ** 2).sum(axis=0).sum(axis=0)

    def weight_hh_l0(
        self,
        ext,
        module: RNN,
        grad_inp: Tuple[Tensor],
        grad_out: Tuple[Tensor],
        backproped: Tensor,
    ) -> Tensor:
        """Returns diagonal of GGN.

        Args:
            ext: extension
            module: module through which to backpropagate
            grad_inp: input gradients
            grad_out: output gradients
            backproped: backpropagated information

        Returns:
            diagonal
        """
        JS = self.derivatives.weight_hh_l0_jac_t_mat_prod(
            module, grad_inp, grad_out, backproped, sum_batch=False
        )
        return (JS ** 2).sum(axis=0).sum(axis=0)
