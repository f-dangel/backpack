"""Contains base class for ``SqrtGGN{Exact, MC}`` module extensions."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Tuple, Union

from torch import Tensor
from torch.nn import Module

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.extensions.mat_to_mat_jac_base import MatToJacMat

if TYPE_CHECKING:
    from backpack.extensions.secondorder.sqrt_ggn import SqrtGGNExact, SqrtGGNMC


class SqrtGGNBaseModule(MatToJacMat):
    """Base module extension for ``SqrtGGN{Exact, MC}``."""

    def __init__(self, derivatives: BaseDerivatives, params: List[str] = None):
        """Store parameter names and derivatives.

        Sets up methods that extract the GGN/Fisher matrix square root for the
        passed parameters, unless these methods are overwritten by a child class.

        Args:
            derivatives: derivatives object.
            params: List of parameter names. Defaults to None.
        """
        if params is not None:
            for param_str in params:
                if not hasattr(self, param_str):
                    setattr(self, param_str, self._make_param_function(param_str))

        super().__init__(derivatives, params=params)

    def _make_param_function(
        self, param_str: str
    ) -> Callable[
        [Union[SqrtGGNExact, SqrtGGNMC], Module, Tuple[Tensor], Tuple[Tensor], Tensor],
        Tensor,
    ]:
        """Create a function that computes the GGN/Fisher square root for a parameter.

        Args:
            param_str: name of parameter

        Returns:
            Function that computes the GGN/Fisher matrix square root.
        """

        def param_function(
            ext: Union[SqrtGGNExact, SqrtGGNMC],
            module: Module,
            g_inp: Tuple[Tensor],
            g_out: Tuple[Tensor],
            backproped: Tensor,
        ) -> Tensor:
            """Calculate the GGN/Fisher matrix square root with the derivatives object.

            Args:
                ext: extension that is used
                module: module that performed forward pass
                g_inp: input gradient tensors
                g_out: output gradient tensors
                backproped: Backpropagated quantities from second-order extension.

            Returns:
                GGN/Fisher matrix square root.
            """
            return self.derivatives.param_mjp(
                param_str,
                module,
                g_inp,
                g_out,
                backproped,
                sum_batch=False,
                subsampling=ext.get_subsampling(),
            )

        return param_function
