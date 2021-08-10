"""Contains base class for second order extensions."""
from typing import List, Tuple, Union

from torch import Tensor
from torch.nn import Module

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.module_extension import ModuleExtension


class MatToJacMat(ModuleExtension):
    """Base class for backpropagation of matrices by multiplying with Jacobians."""

    def __init__(self, derivatives: BaseDerivatives, params: List[str] = None):
        """Initialization.

        Args:
            derivatives: class containing derivatives
            params: list of parameter names
        """
        super().__init__(params)
        self.derivatives = derivatives

    def backpropagate(
        self,
        ext: BackpropExtension,
        module: Module,
        grad_inp: Tuple[Tensor],
        grad_out: Tuple[Tensor],
        backproped: Union[List[Tensor], Tensor],
    ) -> Union[List[Tensor], Tensor]:
        """Propagates second order information back.

        Args:
            ext: BackPACK extension
            module: module through which to perform backpropagation
            grad_inp: input gradients
            grad_out: output gradients
            backproped: backpropagation information

        Returns:
            derivative wrt input
        """
        subsampling = ext.get_subsampling()

        if isinstance(backproped, list):
            return [
                self.derivatives.jac_t_mat_prod(
                    module, grad_inp, grad_out, M, subsampling=subsampling
                )
                for M in backproped
            ]
        else:
            return self.derivatives.jac_t_mat_prod(
                module, grad_inp, grad_out, backproped, subsampling=subsampling
            )
