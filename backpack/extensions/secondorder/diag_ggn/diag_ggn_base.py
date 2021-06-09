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
    ):
        """Initialization.

        Creates a method named after parameter for each parameter. Checks if that
        method is implemented, so a child class can implement a more efficient version.

        Args:
            derivatives: class containing derivatives
            params: list of parameter names
        """
        if params is not None:
            for param in params:
                if not hasattr(self, param):
                    setattr(self, param, self._make_param_method(param))
        super().__init__(derivatives, params=params)

    def _make_param_method(
        self, param: str
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
            JS: Tensor = getattr(self.derivatives, f"{param}_jac_t_mat_prod")(
                module, grad_inp, grad_out, backproped, sum_batch=False
            )
            return (JS ** 2).sum(axis=0).sum(axis=0)

        return _param
