"""Contains base class and extensions for losses used by ``SqrtGGN{Exact, MC}``."""
from functools import partial
from typing import Any, Callable, Tuple

from torch import Tensor
from torch.nn import Module

from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.extensions.secondorder.hbp import LossHessianStrategy
from backpack.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNBaseLossModule(SqrtGGNBaseModule):
    """Base class for losses used by ``SqrtGGN{Exact, MC}``."""

    # TODO Replace Any with Union[SqrtGGNExact, SqrtGGNMC]
    # WAITING Deprecation of python3.6 (cyclic imports caused by annotations)
    def backpropagate(
        self,
        ext: Any,
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
        """
        hess_func = self.make_loss_hessian_func(ext)

        return hess_func(module, grad_inp, grad_out)

    # TODO Replace Any with Union[SqrtGGNExact, SqrtGGNMC]
    # WAITING Deprecation of python3.6 (cyclic imports caused by annotations)
    def make_loss_hessian_func(
        self, ext: Any
    ) -> Callable[[Module, Tuple[Tensor], Tuple[Tensor]], Tensor]:
        """Return a function that evaluates the backpropagated Hessian's representation.

        Args:
            ext: BackPACK extension that holds the information which representation
                to generate.

        Returns:
            Function that evaluates the loss Hessian's symmetric factorization.

        Raises:
            ValueError: For invalid strategies to represent the loss Hessian.
        """
        loss_hessian_strategy = ext.loss_hessian_strategy

        if loss_hessian_strategy == LossHessianStrategy.EXACT:
            return self.derivatives.sqrt_hessian
        elif loss_hessian_strategy == LossHessianStrategy.SAMPLING:
            mc_samples = ext.get_num_mc_samples()
            return partial(
                self.derivatives.sqrt_hessian_sampled,
                mc_samples=mc_samples,
            )
        else:
            raise ValueError(
                "Unknown hessian strategy {}".format(loss_hessian_strategy)
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
