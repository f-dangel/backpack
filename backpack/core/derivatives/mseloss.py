"""NLL extention for Mean Square Error Loss."""
from abc import ABC
from math import sqrt
from typing import List, Tuple

from torch import Size, Tensor, eye, ones, tensor
from torch.distributions import Normal
from torch.nn import MSELoss

from backpack.core.derivatives.nll_base import NLLLossDerivatives


class MSELossDerivatives(NLLLossDerivatives, ABC):
    """Partial derivatives for mean square error loss.

    We only support 2D tensors.

    For `X : [n, d]` and `Y : [n, d]`, if `reduce=sum`, the MSE computes
    `∑ᵢ₌₁ⁿ ‖X[i,∶] − Y[i,∶]‖²`. If `reduce=mean`, the result is divided by `nd`.
    The square root Hessian can be sampled from a Gaussian distribution
    with a mean of 0 and a variance of √2."""

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

    def _verify_support(self, module: MSELoss):
        self._check_input_dims(module)

    def _make_distribution(self, subsampled_input: Tensor):
        """Make the sampling distribution for the NLL loss form of MSE.

        The log probabiity of the Gaussian distribution is proportional to
        ¹/₍₂𝜎²₎∑ᵢ₌₁ⁿ (xᵢ−𝜇)². Because MSE = ∑ᵢ₌₁ⁿ(Yᵢ−Ŷᵢ)², this is
        equivalent for samples drawn from a Gaussian distribution with
        mean of the subsampled input and variance √0.5.

        Args:
            subsampled_input: input after subsampling

        Returns:
            torch.distributions Normal distribution with mean of
        the subsampled input and variance √0.5
        """
        return Normal(subsampled_input, tensor(sqrt(0.5)).to(subsampled_input.device))

    def _check_input_dims(self, module: MSELoss):
        """Raises an exception if the shapes of the input are not supported."""
        if not len(module.input0.shape) == 2:
            raise ValueError("Only 2D inputs are currently supported for MSELoss.")

    def hessian_is_psd(self) -> bool:
        """Return whether cross-entropy loss Hessian is positive semi-definite.

        Returns:
            True
        """
        return True

    @staticmethod
    def _get_mean_normalization(input: Tensor) -> int:
        return input.numel()

    def compute_sampled_grads(
        self, subsampled_input: Tensor, mc_samples: int, use_autograd: bool = False
    ):
        """Custom method to overwrite gradient computation for MeanSquareError Loss.

        Because MSE = ∑ᵢ₌₁ⁿ(Yᵢ−Ŷᵢ)², the gradient is 2∑ᵢ₋₁ⁿ(Yᵢ−Ŷᵢ).

        Args:
            subsampled_input: input after subsampling
            mc_samples: number of samples
            use_autograd: boolean to use NLL version of compute_sampled_grads for testing

        Returns:
            sampled gradient
        """
        if use_autograd:
            return super().compute_sampled_grads(
                subsampled_input, mc_samples, use_autograd
            )

        dist = self._make_distribution(subsampled_input)
        samples = dist.sample(sample_shape=Size([mc_samples]))
        subsampled_input_expanded = subsampled_input.unsqueeze(0).expand(
            mc_samples, -1, -1
        )

        return 2 * (samples - subsampled_input_expanded)
