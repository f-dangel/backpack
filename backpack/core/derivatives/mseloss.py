"""Derivatives of the MSE Loss."""

from math import sqrt
from typing import List, Tuple

from torch import Tensor, eye, normal, ones
from torch.nn import MSELoss

from backpack.core.derivatives.basederivatives import BaseLossDerivatives


class MSELossDerivatives(BaseLossDerivatives):
    """Derivatives of the MSE Loss.

    We only support 2D tensors.

    For `X : [n, d]` and `Y : [n, d]`, if `reduce=sum`, the MSE computes
    `∑ᵢ₌₁ⁿ ‖X[i,∶] − Y[i,∶]‖²`. If `reduce=mean`, the result is divided by `nd`.
    """

    def _sqrt_hessian(
        self,
        module: MSELoss,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:  # noqa: D102
        self.check_input_dims(module)

        input0: Tensor = module.input0
        N, D = input0.shape
        N_active = N if subsampling is None else len(subsampling)

        scale = sqrt(2)
        if module.reduction == "mean":
            scale /= sqrt(input0.numel())

        sqrt_H_diag = scale * ones(D, device=input0.device, dtype=input0.dtype)
        sqrt_H = sqrt_H_diag.diag().unsqueeze(1).expand(-1, N_active, -1)

        return sqrt_H

    def _sqrt_hessian_sampled(
        self,
        module: MSELoss,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mc_samples: int = 1,
        subsampling: List[int] = None,
    ) -> Tensor:
        self.check_input_dims(module)

        input0: Tensor = module.input0
        N, D = input0.shape
        N_active = N if subsampling is None else len(subsampling)
        samples = normal(
            0,
            1,
            size=[mc_samples, N_active, D],
            device=input0.device,
            dtype=input0.dtype,
        )
        samples *= sqrt(2) / sqrt(mc_samples)

        if module.reduction == "mean":
            samples /= sqrt(input0.numel())

        return samples

    def _sum_hessian(self, module, g_inp, g_out):
        """The Hessian, summed across the batch dimension.

        Args:
            module: (torch.nn.MSELoss) module
            g_inp: Gradient of loss w.r.t. input
            g_out: Gradient of loss w.r.t. output

        Returns: a `[D, D]` tensor of the Hessian, summed across batch

        """
        self.check_input_dims(module)

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

    def check_input_dims(self, module):
        """Raises an exception if the shapes of the input are not supported."""
        if not len(module.input0.shape) == 2:
            raise ValueError("Only 2D inputs are currently supported for MSELoss.")

    def hessian_is_psd(self):
        return True
