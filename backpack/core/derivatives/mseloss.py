"""Derivatives of the MSE Loss."""

from math import sqrt
from typing import List, Tuple

from torch import Size, Tensor, eye, ones, tensor
from torch.distributions import Normal
from torch.nn import MSELoss

from backpack.core.derivatives.nll_base import NLLLossDerivatives


class MSELossDerivatives(NLLLossDerivatives):
    """Derivatives of ``MSELoss``.

    We only support 2D tensors.

    For `X : [n, d]` and `Y : [n, d]`, if `reduce=sum`, the MSE computes
    `∑ᵢ₌₁ⁿ ‖X[i,∶] − Y[i,∶]‖²`. If `reduce=mean`, the result is divided by `nd`.

    ``MSELoss`` is a negative log-likelihood of a Gaussian with mean corresponding
    to the module input and constant standard deviation √0.5.
    """

    def __init__(self, use_autograd: bool = False):
        """Initialization for MSE loss derivative.

        Args:
            use_autograd: Compute gradients with autograd (rather than manual)
                Defaults to ``False`` (manual computation).
        """
        super().__init__(use_autograd=use_autograd)

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
        """We only support 2D tensors."""
        self._check_input_dims(module)

    def _make_distribution(self, subsampled_input: Tensor) -> Normal:
        """Create the likelihood distribution whose NLL is the MSE.

        The log probability of the Gaussian distribution is proportional to
        ¹/₍₂𝜎²₎∑ᵢ₌₁ⁿ (xᵢ−𝜇)². Because MSE = ∑ᵢ₌₁ⁿ(Yᵢ−Ŷᵢ)², this is
        equivalent for samples drawn from a Gaussian distribution with
        mean of the subsampled input and standard deviation √0.5.

        Args:
            subsampled_input: input after subsampling

        Returns:
            Normal distribution for targets | inputs
        """
        return Normal(
            subsampled_input, tensor(sqrt(0.5), device=subsampled_input.device)
        )

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

    def _compute_sampled_grads_manual(
        self, subsampled_input: Tensor, mc_samples: int
    ) -> Tensor:
        """Manually compute gradients from sampled targets.

        Because MSE = ∑ᵢ₌₁ⁿ(Yᵢ−Ŷᵢ)², the gradient is 2∑ᵢ₋₁ⁿ(Yᵢ−Ŷᵢ).

        Args:
            subsampled_input: input after subsampling
            mc_samples: number of samples

        Returns:
            Gradient samples
        """
        dist = self._make_distribution(subsampled_input)
        samples = dist.sample(sample_shape=Size([mc_samples]))
        subsampled_input_expanded = subsampled_input.unsqueeze(0).expand(
            mc_samples, -1, -1
        )

        return 2 * (samples - subsampled_input_expanded)
