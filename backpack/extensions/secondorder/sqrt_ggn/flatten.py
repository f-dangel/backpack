"""Contains extensions for the flatten layer used by ``SqrtGGN{Exact, MC}``."""
from typing import Any, Tuple

from torch import Tensor
from torch.nn import Module

from backpack.core.derivatives.flatten import FlattenDerivatives
from backpack.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNFlatten(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.Flatten`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Flatten`` module."""
        super().__init__(FlattenDerivatives())

    # TODO Replace Any with Union[SqrtGGNExact, SqrtGGNMC]
    # WAITING Deprecation of python3.6 (cyclic imports caused by annotations)
    def backpropagate(
        self,
        ext: Any,
        module: Module,
        grad_inp: Tuple[Tensor],
        grad_out: Tuple[Tensor],
        backproped: Tensor,
    ) -> Tensor:
        """Backpropagate only if flatten created a node in the computation graph.

        Otherwise, the backward hook will not be called at the right stage and
        no action must be performed.

        Args:
            ext: BackPACK extension calling out to the module extension.
            module: Module that performed the forward pass.
            grad_inp: Gradients w.r.t. the module inputs.
            grad_out: Gradients w.r.t. the module outputs.
            backproped: Backpropagated symmetric factorization of the loss Hessian
                from the child module.

        Returns:
            Symmetric loss Hessian factorization, backpropagated through the module.
        """
        if self.derivatives.is_no_op(module):
            return backproped
        else:
            return super().backpropagate(ext, module, grad_inp, grad_out, backproped)
