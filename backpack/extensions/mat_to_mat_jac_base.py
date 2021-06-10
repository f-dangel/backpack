"""Contains base class for second order extensions."""
from typing import List, Tuple, Union

from torch import Tensor
from torch.nn import Module

from ..core.derivatives.basederivatives import BaseDerivatives, BaseParameterDerivatives
from .module_extension import ModuleExtension


class MatToJacMat(ModuleExtension):
    """Base class for backpropagation of matrices by multiplying with Jacobians."""

    def __init__(
        self,
        derivatives: Union[BaseDerivatives, BaseParameterDerivatives],
        params: List[str] = None,
    ):
        """Initialization.

        Args:
            derivatives: class containing derivatives
            params: list of parameter names
        """
        super().__init__(params)
        self.derivatives = derivatives

    def backpropagate(
        self,
        ext: ModuleExtension,
        module: Module,
        grad_inp: Tuple[Tensor],
        grad_out: Tuple[Tensor],
        backproped: Union[List[Tensor], Tensor],
    ) -> Union[List[Tensor], Tensor]:
        """Propagates second order information back.

        Args:
            ext: extension
            module: module through which to perform backpropagation
            grad_inp: input gradients
            grad_out: output gradients
            backproped: backpropagation information

        Returns:
            derivative wrt input
        """
        if isinstance(backproped, list):
            M_list: List[Tensor] = [
                self.derivatives.jac_t_mat_prod(module, grad_inp, grad_out, M)
                for M in backproped
            ]
            return M_list
        else:
            return self.derivatives.jac_t_mat_prod(
                module, grad_inp, grad_out, backproped
            )
