"""Partial derivative bases for NLL losses."""
from math import sqrt
from typing import List, Tuple

from torch import Size, Tensor, enable_grad, stack
from torch.autograd import Variable, grad
from torch.nn import Module

from backpack.core.derivatives.basederivatives import BaseLossDerivatives
from backpack.utils.subsampling import subsample


class NLLLossDerivatives(BaseLossDerivatives):
    """Partial derivative bases for NLL loss.

    For loss functions that can be expressed as a Negative Log Likelihood (NLL), the
    Maximum a-Posteriori (MAP) estimate of the network parameters can be written as
    𝜃ₘₐₚ = argmin_𝜃 (𝑟(𝜃)+𝑙(xₙ,yₙ;𝜃)) where 𝑟 is the regularizer and 𝑙 is the loss
    function, defined here as 𝑙(xₙ,yₙ;𝜃)= −log p(yₙ | f_𝜃(xₙ))."""

    def _sqrt_hessian_sampled(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mc_samples: int = 1,
        subsampling: List[int] = None,
        use_dist: bool = False,
    ) -> Tensor:
        """Method to approximate the square root Hessian through Monte Carlo sampling.

        For a given loss, either _make_distribution or compute_sampled_grads must be
        implemented.

        Optionally, it is possible in _verify_support to specify checks to perform
        on the input module before calculating the loss gradient, to verify that
        it's parameters are supported for that particular loss.

        For use in mean mode, _get_mean_normalization must be implemented.

        Args:
            module: loss module
            g_inp: Gradient of loss w.r.t. input
            g_out: Gradient of loss w.r.t. output
            mc_samples: number of Monte Carlo samples to take
            subsampling: Indices of samples that are sliced along the dimension.
            Defaults to ``None`` (use all samples).

        Returns:
            Monte Carlo sampled gradients
        """
        self._verify_support(module)
        subsampled_input = subsample(module.input0, subsampling=subsampling)
        sqrt_hessian = self.compute_sampled_grads(
            subsampled_input, mc_samples, use_dist
        ) / sqrt(mc_samples)
        if module.reduction == "mean":
            sqrt_hessian /= sqrt(self._get_mean_normalization(module.input0))
        return sqrt_hessian

    def _verify_support(self, module: Module):
        """Any checks to be performed to verify module support."""
        raise NotImplementedError

    def compute_sampled_grads(
        self, subsampled_input: Tensor, mc_samples: int, use_dist: bool = False
    ):
        """Method to create the sampled gradients.

        This method returns the gradient of the loss with respect
        to each of the randomly drawn samples. To use this function, the user must implement
        the functions _make_distribution.

        By default, this will compute gradients for samples of the likelihood distribution
        with autograd. This function can be overwritten if the gradient is known analytically.
        In this case, the method should return the gradient of the loss with respect to the
        subsampled input for each of the Monte Carlo samples

        Args:
            subsampled_input: input after subsampling
            mc_samples: number of samples
            use_dist: boolean to use NLL version of compute_sampled_grads for testing

        Returns:
            sampled gradient of shape [mc_samples, *subsampled_input.shape]
        """
        subsampled_input = Variable(subsampled_input, requires_grad=True)
        with enable_grad():
            gradient = []
            dist = self._make_distribution(subsampled_input)
            y_tilde = dist.sample(sample_shape=Size([mc_samples]))
            loss_tilde = -dist.log_prob(y_tilde)
            for m in range(mc_samples):
                gradient.append(
                    grad(
                        loss_tilde[m].sum(),
                        subsampled_input,
                        retain_graph=True,
                    )[0]
                )
        return stack(gradient)

    def _make_distribution(self, subsampled_input: Tensor):
        """Create the negative log likelihood distribution.

        This should be in the form of a torch.Distributions object, such that
        the desired loss 𝑙(xₙ,yₙ;𝜃)= −log p(yₙ | f_𝜃(xₙ)).

        This torch.Distributions object must include the functions
        sample(sample_shape=torch.Size([])) and log_prob(value).

        Args:
            subsampled_input: input after subsampling

        Returns:
            torch.Distributions object for the likelihood p(y | xₙ, 𝜃)
        """
        raise NotImplementedError

    @staticmethod
    def _get_mean_normalization(input: Tensor) -> int:
        """Normalization factor for mean mode.

        The number C in loss = 1 / C * ∑ᵢ lossᵢ.

        If used in mean mode, the normalization factor must be provided.

        Args:
            input: input to the layer

        Returns:
            normalization factor for mean mode
        """
        raise NotImplementedError
