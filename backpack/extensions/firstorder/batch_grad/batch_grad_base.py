"""Calculates the batch_grad derivative."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Tuple

from torch import Tensor
from torch.nn import Module

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.utils.subsampling import subsample

if TYPE_CHECKING:
    from backpack.extensions.firstorder import BatchGrad


class BatchGradBase(FirstOrderModuleExtension):
    """Calculates the batch_grad derivative.

    Passes the calls for the parameters to the derivatives class.
    Implements functions with method names from params.

    If child class wants to overwrite these methods
    - for example to support an additional external module -
    it can do so using the interface for parameter "param1"::

        param1(ext, module, g_inp, g_out, bpQuantities):
            return batch_grads

    In this case, the method is not overwritten by this class.
    """

    def __init__(
        self, derivatives: BaseParameterDerivatives, params: List[str]
    ) -> None:
        """Initializes all methods.

        If the param method has already been defined, it is left unchanged.

        Args:
            derivatives: Derivatives object used to apply parameter Jacobians.
            params: List of parameter names.
        """
        self._derivatives = derivatives
        for param_str in params:
            if not hasattr(self, param_str):
                setattr(self, param_str, self._make_param_function(param_str))
        super().__init__(params=params)

    def _make_param_function(
        self, param_str: str
    ) -> Callable[[BatchGrad, Module, Tuple[Tensor], Tuple[Tensor], None], Tensor]:
        """Creates a function that calculates batch_grad w.r.t. param.

        Args:
            param_str: Parameter name.

        Returns:
            Function that calculates batch_grad wrt param
        """

        def param_function(
            ext: BatchGrad,
            module: Module,
            g_inp: Tuple[Tensor],
            g_out: Tuple[Tensor],
            bpQuantities: None,
        ) -> Tensor:
            """Calculates batch_grad with the help of derivatives object.

            Args:
                ext: extension that is used
                module: module that performed forward pass
                g_inp: input gradient tensors
                g_out: output gradient tensors
                bpQuantities: additional quantities for second order

            Returns:
                Scaled individual gradients
            """
            subsampling = ext.get_subsampling()
            batch_axis = 0

            return self._derivatives.param_mjp(
                param_str,
                module,
                g_inp,
                g_out,
                subsample(g_out[0], dim=batch_axis, subsampling=subsampling),
                sum_batch=False,
                subsampling=subsampling,
            )

        return param_function
