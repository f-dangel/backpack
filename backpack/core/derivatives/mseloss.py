"""NLL extention for Mean Square Error Loss."""
from abc import ABC


from typing import List, Tuple
from torch import Tensor, eye, mul, ones, zeros
from torch.distributions import MultivariateNormal
from torch.nn import MSELoss

from backpack.core.derivatives.nll_base import NLLLossDerivatives
from math import sqrt


class MSELossDerivatives(NLLLossDerivatives, ABC):
    """Partial derivatives for mean square erro loss.

    This comes from the Gaussian distribution.
    """

    def _sqrt_hessian(
        self,
        module: MSELoss,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:  # noqa: D102
        self._check_input_dims(module)

        input0: Tensor = module.input0
        N, D = input0.shape
        N_active = N if subsampling is None else len(subsampling)

        scale = sqrt(2)
        if module.reduction == "mean":
            scale /= sqrt(input0.numel())

        sqrt_H_diag = scale * ones(D, device=input0.device, dtype=input0.dtype)
        sqrt_H = sqrt_H_diag.diag().unsqueeze(1).expand(-1, N_active, -1)

        return sqrt_H

    def _sum_hessian(self, module, g_inp, g_out):
        """The Hessian, summed across the batch dimension.

        Args:
            module: (torch.nn.MSELoss) module
            g_inp: Gradient of loss w.r.t. input
            g_out: Gradient of loss w.r.t. output

        Returns: a `[D, D]` tensor of the Hessian, summed across batch

        """
        self._check_input_dims(module)

        N, D = module.input0.shape
        H = 2 * eye(D, device=module.input0.device)

        if module.reduction == "sum":
            H *= N
        elif module.reduction == "mean":
            H /= D

        return H

    def _make_hessian_mat_prod(self, module, g_inp, g_out):
        """Multiplication of the input Hessian with a matrix."""

        def hessian_mat_prod(mat):
            Hmat = 2 * mat
            if module.reduction == "mean":
                Hmat /= module.input0.numel()
            return Hmat

        return hessian_mat_prod

    def _checks(self, module):
        self._check_input_dims(module)

    def _make_distribution(self, subsampled_input, mc_samples):
        return MultivariateNormal(
            zeros(len(subsampled_input) * len(subsampled_input[0])),
            mul(eye(len(subsampled_input) * len(subsampled_input[0])), 2),
        )

    def _check_input_dims(self, module):
        """Raises an exception if the shapes of the input are not supported."""
        if not len(module.input0.shape) == 2:
            raise ValueError("Only 2D inputs are currently supported for MSELoss.")

    def hessian_is_psd(self) -> bool:
        """Return whether cross-entropy loss Hessian is positive semi-definite.

        Returns:
            True
        """
        return True
