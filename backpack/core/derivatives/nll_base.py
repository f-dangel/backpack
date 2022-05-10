"""Partial derivative bases for NLL losses."""
from math import sqrt
from typing import List, Tuple

from torch import Tensor, stack
from torch.autograd import grad
from torch.distributions import Distribution
from torch.nn import Module

from backpack.core.derivatives.basederivatives import BaseLossDerivatives
from backpack.utils.subsampling import subsample


class NLLLossDerivatives(BaseLossDerivatives):
    """Partial derivative bases for NLL loss.

    NLL Loss functions can be expressed as a Negative Log Likelihood (NLL)
    𝑙(xₙ,yₙ;𝜃)= −log p(yₙ | f_𝜃(xₙ)).
    """

    def __init__(self, use_autograd: bool = True):
        """Initialization.

        Args:
            use_autograd: compute gradients with autograd (rather than manual).
                Default: ``True`` This argument is used to test automated and manual
                versions of sampled gradient computation.
        """
        self.use_autograd = use_autograd

    def _sqrt_hessian_sampled(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mc_samples: int = 1,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Approximate the Hessian square root through Monte-Carlo sampling.

        With use_autograd true, _make_distribution must be implemented for the
        loss function. If use_autograd is False, _compute_sampled_grads_manual
        must be implemented to calculate the MC samples.

        Optionally, it is possible in _verify_support to specify checks to perform
        on the input module before calculating the loss gradient, to verify that
        it's parameters are supported for that particular loss.

        For use in mean mode, _get_mean_normalization must be implemented.

        Args:
            module: loss module.
            g_inp: Gradient of loss w.r.t. input
            g_out: Gradient of loss w.r.t. output
            mc_samples: number of Monte Carlo samples to take
            subsampling: Indices of samples that are sliced along the dimension

        Returns:
            Approximate Hessian square root
        """
        self._verify_support(module)
        subsampled_input = subsample(module.input0, subsampling=subsampling)
        sqrt_hessian = self.compute_sampled_grads(subsampled_input, mc_samples) / sqrt(
            mc_samples
        )
        if module.reduction == "mean":
            sqrt_hessian /= sqrt(self._get_mean_normalization(module.input0))
        return sqrt_hessian

    def _verify_support(self, module: Module):
        """Verification that the module is supported for the loss function.

        Args:
            module: loss module

        Raises:
            NotImplementedError: if module verification has not been provided for the loss
        """
        raise NotImplementedError

    def compute_sampled_grads(
        self, subsampled_input: Tensor, mc_samples: int
    ) -> Tensor:
        """Compute gradients with targets drawn from the likelihood.

        If use_autograd is True, use _compute_sampled_grads_autograd.
        Otherwise, _compute_sampled_grads_manual will be used.

        Args:
            subsampled_input: input after subsampling
            mc_samples: number of samples

        Returns:
            Sampled gradients of shape [mc_samples, *subsampled_input.shape]
        """
        grad_func = (
            self._compute_sampled_grads_autograd
            if self.use_autograd
            else self._compute_sampled_grads_manual
        )
        return grad_func(subsampled_input, mc_samples)

    def _compute_sampled_grads_autograd(
        self, subsampled_input: Tensor, mc_samples: int
    ) -> Tensor:
        """Compute gradients for samples of the likelihood distribution with autograd.

        _make_distribution must be implemented for this function to work.

        Args:
            subsampled_input: input after subsampling
            mc_samples: number of samples

        Returns:
            Sampled gradients of shape [mc_samples, *subsampled_input.shape]
        """
        subsampled_input.requires_grad = True
        gradients = []

        dist = self._make_distribution(subsampled_input)
        self._check_distribution_shape(dist, subsampled_input)

        for _ in range(mc_samples):
            y_tilde = dist.sample()
            loss_tilde = -dist.log_prob(y_tilde).sum()
            gradients.append(grad(loss_tilde, subsampled_input)[0])

        return stack(gradients)

    def _compute_sampled_grads_manual(
        self, subsampled_input: Tensor, mc_samples: int
    ) -> Tensor:
        """Compute gradients for samples of the likelihood distribution manually.

        This function can be used instead of _compute_sampled_grads_autograd if the gradient
        is known analytically.

        Args:
            subsampled_input: input after subsampling
            mc_samples: number of samples

        Raises:
            NotImplementedError: if manual sampled gradients not implemented
        """
        raise NotImplementedError("Manual sampled gradients not implemented.")

    def _make_distribution(self, subsampled_input: Tensor) -> Distribution:
        """Create the likelihood distribution.

        This should be in the form of a torch.Distributions object for p, such that
        the desired loss 𝑙(xₙ,yₙ;𝜃)= −log p(yₙ | f_𝜃(xₙ)).

        This torch.Distributions object must include the functions
        sample(sample_shape=torch.Size([])) and log_prob(value).

        Args:
            subsampled_input: input after subsampling

        Raises:
            NotImplementedError: if the distribution function has not been provided
        """
        raise NotImplementedError

    @staticmethod
    def _get_mean_normalization(input: Tensor) -> int:
        """Normalization factor for mean mode.

        The number C in loss = 1 / C * ∑ᵢ lossᵢ.

        Args:
            input: input to the layer

        Raises:
            NotImplementedError: if the mean normalization has not been provided
        """
        raise NotImplementedError

    @staticmethod
    def _check_distribution_shape(dist: Distribution, subsampled_input: Tensor):
        """Verify shape of sampled targets.

        The distribution returned by _make_distribution must sample tensors
        with the same shape as subsampled_input.

        Args:
            dist: torch.Distributions object for the likelihood p(y | xₙ, 𝜃)
                as returned by _make_distribution
            subsampled_input: input after subsampling

        Raises:
            ValueError: if dist.sample() does not return an object of the same
                shape as subsampled_input
        """
        if dist.sample().shape != subsampled_input.shape:
            raise ValueError("Sample does not have same shape as subsampled_input.")
