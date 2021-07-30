"""Contains DiagGGN base class."""
from typing import Callable, List, Tuple, Union

from torch import Tensor
from torch.nn import Module

from backpack.core.derivatives.basederivatives import (
    BaseDerivatives,
    BaseParameterDerivatives,
)
from backpack.extensions.mat_to_mat_jac_base import MatToJacMat
from backpack.extensions.module_extension import ModuleExtension


class DiagGGNBaseModule(MatToJacMat):
    """Base class for DiagGGN extension."""

    def __init__(
        self,
        derivatives: Union[BaseDerivatives, BaseParameterDerivatives],
        params: List[str] = None,
        sum_batch: bool = None,
    ):
        """Initialization.

        If params and sum_batch is provided:
        Creates a method named after parameter for each parameter. Checks if that
        method is implemented, so a child class can implement a more efficient version.

        Args:
            derivatives: class containing derivatives
            params: list of parameter names. Defaults to None.
            sum_batch: Specifies whether the created method sums along batch axis.
                For BatchDiagGGNModule should be False.
                For DiagGGNModule should be True.
                Defaults to None.
        """
        if params is not None and sum_batch is not None:
            for param in params:
                if not hasattr(self, param):
                    setattr(self, param, self._make_param_method(param, sum_batch))
        super().__init__(derivatives, params=params)

    def _make_param_method(
        self, param_str: str, sum_batch: bool
    ) -> Callable[
        [ModuleExtension, Module, Tuple[Tensor], Tuple[Tensor], Tensor], Tensor
    ]:
        def _param(
            ext: ModuleExtension,
            module: Module,
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
            axis: Tuple[int] = (0, 1) if sum_batch else (0,)
            return (
                self.derivatives.param_mjp(
                    param_str, module, grad_inp, grad_out, backproped, sum_batch=False
                )
                ** 2
            ).sum(axis=axis)

        return _param
