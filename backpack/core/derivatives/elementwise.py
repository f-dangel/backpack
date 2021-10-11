"""Base class for more flexible Jacobians/Hessians of activation functions."""

from typing import List, Tuple

from torch import Tensor, einsum
from torch.nn import Module

from backpack.core.derivatives.basederivatives import BaseDerivatives


class ElementwiseDerivatives(BaseDerivatives):
    """Extended autodifferentiation functionality for element-wise activations.

    Element-wise functions have diagonal Jacobians/Hessians, since `output[i]`
    only depends on `input[i]`, and not on any `input[j ≠ i]`.

    The forward pass is `output[i] = f(input[i]) ∀ i`, where `f` denotes the
    activation function.

    Notes:
        - Methods that need to be implemented:
          - Required: `df`.
          - If the activation is piece-wise linear: `hessian_is_zero`, else `d2f`.
    """

    def df(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ):
        """Elementwise first derivative.

        Args:
            module: PyTorch activation module.
            g_inp: Gradients of the module w.r.t. its inputs.
            g_out: Gradients of the module w.r.t. its outputs.
            subsampling: Indices of active samples. ``None`` means all samples.

        Returns:
            Tensor containing the derivatives `f'(input[i]) ∀ i`.
        """
        raise NotImplementedError("First derivatives not implemented")

    def d2f(self, module, g_inp, g_out):
        """Elementwise second derivative.

        Only needs to be implemented for non piece-wise linear functions.

        Args:
            module (torch.nn.Module): PyTorch activation module.
            g_inp ([torch.Tensor]): Gradients of the module w.r.t. its inputs.
            g_out ([torch.Tensor]): Gradients of the module w.r.t. its outputs.

        Returns:
            (torch.Tensor): Tensor containing the derivatives `f''(input[i]) ∀ i`.
        """
        raise NotImplementedError("Second derivatives not implemented")

    def hessian_diagonal(self, module, g_inp, g_out):
        """Return `∂²output[i] / ∂input[i]²`.

        Notes:
            - Only required if `hessian_is_diagonal` returns `True`.

        Args:
            module (torch.nn.Module): PyTorch activation module.
            g_inp ([torch.Tensor]): Gradients of the module w.r.t. its inputs.
            g_out ([torch.Tensor]): Gradients of the module w.r.t. its outputs.
        """
        return self.d2f(module, g_inp, g_out) * g_out[0]

    def hessian_is_diagonal(self, module):
        """Elementwise activation function Hessians are diagonal.

        Returns:
            bool: True
        """
        return True

    def _jac_t_mat_prod(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        df_elementwise = self.df(module, g_inp, g_out, subsampling=subsampling)
        return einsum("...,v...->v...", df_elementwise, mat)

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        return self.jac_t_mat_prod(module, g_inp, g_out, mat)

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        N = module.input0.size(0)
        df_flat = self.df(module, g_inp, g_out).reshape(N, -1)
        return einsum("ni,nj,ij->ij", df_flat, df_flat, mat) / N

    def _residual_mat_prod(self, module, g_inp, g_out, mat):
        residual = self.d2f(module, g_inp, g_out) * g_out[0]
        return einsum("...,v...->v...", residual, mat)
