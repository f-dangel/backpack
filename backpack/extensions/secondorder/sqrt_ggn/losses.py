"""Contains base class and extensions for losses used by ``SqrtGGN{Exact, MC}``."""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

from torch import Tensor
from torch.nn import Module

from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.extensions.secondorder.hbp import LossHessianStrategy
from backpack.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule

if TYPE_CHECKING:
    from backpack.extensions.secondorder.sqrt_ggn import SqrtGGNExact, SqrtGGNMC


class SqrtGGNBaseLossModule(SqrtGGNBaseModule):
    """Base class for losses used by ``SqrtGGN{Exact, MC}``."""

    def backpropagate(
        self,
        ext: Union[SqrtGGNExact, SqrtGGNMC],
        module: Module,
        grad_inp: Tuple[Tensor],
        grad_out: Tuple[Tensor],
        backproped: None,
    ) -> Tensor:
        """Initialize the backpropagated quantity.

        Uses the exact loss Hessian square root, or a Monte-Carlo approximation
        thereof.

        Args:
            ext: BackPACK extension calling out to the module extension.
            module: Module that performed the forward pass.
            grad_inp: Gradients w.r.t. the module inputs.
            grad_out: Gradients w.r.t. the module outputs.
            backproped: Backpropagated information. Should be ``None``.

        Returns:
            Symmetric factorization of the loss Hessian w.r.t. the module input.

        Raises:
            NotImplementedError: For invalid strategies to represent the loss Hessian.
        """
        loss_hessian_strategy = ext.get_loss_hessian_strategy()
        subsampling = ext.get_subsampling()

        if loss_hessian_strategy == LossHessianStrategy.EXACT:
            return self.derivatives.sqrt_hessian(
                module, grad_inp, grad_out, subsampling=subsampling
            )
        elif loss_hessian_strategy == LossHessianStrategy.SAMPLING:
            mc_samples = ext.get_num_mc_samples()
            return self.derivatives.sqrt_hessian_sampled(
                module,
                grad_inp,
                grad_out,
                mc_samples=mc_samples,
                subsampling=subsampling,
            )
        else:
            raise NotImplementedError(
                f"Unknown hessian strategy {loss_hessian_strategy}"
            )


class SqrtGGNMSELoss(SqrtGGNBaseLossModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.MSELoss`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.MSELoss`` module."""
        super().__init__(MSELossDerivatives())


class SqrtGGNCrossEntropyLoss(SqrtGGNBaseLossModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.CrossEntropyLoss`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.CrossEntropyLoss`` module."""
        super().__init__(CrossEntropyLossDerivatives())
