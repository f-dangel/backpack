"""Contains VarianceBaseModule."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Tuple

from torch import Tensor
from torch.nn import Module

from backpack.extensions.firstorder.base import FirstOrderModuleExtension

if TYPE_CHECKING:
    from backpack.extensions import Variance
    from backpack.extensions.firstorder.gradient.base import GradBaseModule
    from backpack.extensions.firstorder.sum_grad_squared.sgs_base import SGSBase


class VarianceBaseModule(FirstOrderModuleExtension):
    """Base class for extensions calculating variance."""

    def __init__(
        self,
        params: List[str],
        grad_extension: GradBaseModule,
        sgs_extension: SGSBase,
    ):
        """Initialization.

        Creates a function named after each parameter.

        Args:
            params: list of parameter names
            grad_extension: the extension calculating grad.
            sgs_extension: the extension calculating squared_grad_sum.
        """
        self.grad_ext: GradBaseModule = grad_extension
        self.sgs_ext: SGSBase = sgs_extension
        for param_str in params:
            if not hasattr(self, param_str):
                setattr(self, param_str, self._make_param_function(param_str))
        super().__init__(params=params)

    @staticmethod
    def _variance_from(grad: Tensor, sgs: Tensor, N: int) -> Tensor:
        avgg_squared = (grad / N) ** 2
        avg_gsquared = sgs / N
        return avg_gsquared - avgg_squared

    def _make_param_function(
        self, param: str
    ) -> Callable[[Variance, Module, Tuple[Tensor], Tuple[Tensor], None], Tensor]:
        """Creates a function that calculates variance of grad_batch.

        Args:
            param: name of parameter

        Returns:
            function that calculates variance of grad_batch
        """

        def param_function(
            ext: Variance,
            module: Module,
            g_inp: Tuple[Tensor],
            g_out: Tuple[Tensor],
            bpQuantities: None,
        ) -> Tensor:
            """Calculates variance with the help of derivatives object.

            Args:
                ext: extension that is used
                module: module that performed forward pass
                g_inp: input gradient tensors
                g_out: output gradient tensors
                bpQuantities: additional quantities for second order

            Returns:
                variance of the batch
            """
            return self._variance_from(
                getattr(self.grad_ext, param)(ext, module, g_inp, g_out, bpQuantities),
                getattr(self.sgs_ext, param)(ext, module, g_inp, g_out, bpQuantities),
                g_out[0].shape[0],
            )

        return param_function
