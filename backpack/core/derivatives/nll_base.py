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
    """Base class for partial derivatives of negative log-likelihood losses.

    These loss functions can be expressed as a negative log-likelihood (NLL)
    of targets given the input, 𝑙(fₙ,yₙ)= −log p(yₙ | fₙ) with a likelihood
    distribution p(· | f).
    """

    def __init__(self, use_autograd: bool = True):
        """Initialization.

        Args:
            use_autograd: Compute gradient samples with autograd (rather than manually).
                Default: ``True``. This argument is used to test the non-default
                computation.
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

        If use_autograd is True, _make_distribution must be implemented.
        Otherwise, _compute_sampled_grads_manual must be implemented.

        In mean reduction mode, _get_mean_normalization must be implemented.

        Args:
            module: loss module.
            g_inp: Gradient of loss w.r.t. input
            g_out: Gradient of loss w.r.t. output
            mc_samples: number of Monte Carlo samples to take
            subsampling: Indices of samples that are sliced along the dimension

        Returns:
            Approximate Hessian square root. Has shape [mc_samples,
            subsampled_input.shape].
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
        """Verify that the module hyperparameters are supported.

        Args:
            module: loss module

        Raises:
            NotImplementedError: If the module has unsupported hyperparameters.
        """
        raise NotImplementedError

    def compute_sampled_grads(
        self, subsampled_input: Tensor, mc_samples: int
    ) -> Tensor:
        """Compute gradients with targets drawn from the likelihood p(· | f).

        If use_autograd is True, use _compute_sampled_grads_autograd.
        Otherwise, use _compute_sampled_grads_manual.

        Args:
            subsampled_input: input after subsampling
            mc_samples: number of gradient samples

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
        subsampled_input = self._arrange(subsampled_input.clone().detach())
        subsampled_input.requires_grad = True
        gradients = []

        dist = self._make_distribution(subsampled_input)
        self._check_distribution_shape(dist, subsampled_input)

        for _ in range(mc_samples):
            y_tilde = dist.sample()
            loss_tilde = -dist.log_prob(y_tilde).sum()
            gradients.append(grad(loss_tilde, subsampled_input, retain_graph=True)[0])

        return self._rearrange(stack(gradients))

    def _compute_sampled_grads_manual(
        self, subsampled_input: Tensor, mc_samples: int
    ) -> Tensor:
        """Compute gradients for samples of the likelihood distribution manually.

        This function can be used instead of _compute_sampled_grads_autograd if
        the gradient is known analytically.

        Args:
            subsampled_input: input after subsampling
            mc_samples: number of samples

        Raises:
            NotImplementedError: if manual sampled gradients not implemented
        """
        raise NotImplementedError("Manual sampled gradients not implemented.")

    def _make_distribution(self, subsampled_input: Tensor) -> Distribution:
        """Create the likelihood distribution p(· | f).

        This should be in the form of a torch.Distributions object for p, such that
        the desired loss 𝑙(f, y) α ∑ₙ − log p(yₙ | fₙ).

        Otherwise, the returned object must offer functions to draw samples and to
        evaluate the log-probability.

        Args:
            subsampled_input: input after subsampling

        Raises:
            NotImplementedError: If not implemented.
        """
        raise NotImplementedError

    @staticmethod
    def _get_mean_normalization(input: Tensor) -> int:
        """Return the normalization factor in mean mode.

        The number C in loss = 1 / C * ∑ᵢ lossᵢ.

        Args:
            input: input to the layer

        Raises:
            NotImplementedError: If not implemented
        """
        raise NotImplementedError

    @staticmethod
    def _check_distribution_shape(dist: Distribution, subsampled_input: Tensor):
        """Verify shape of sampled targets y ∼ p(· | f).

        Args:
            dist: Distribution of the likelihood p(y | f), e.g. created by
                _make_distribution.
            subsampled_input: Input after subsampling.

        Raises:
            ValueError: If the target samples have incorrect shape.
        """
        if dist.sample().shape != subsampled_input.shape:
            raise ValueError("Sample does not have same shape as subsampled_input.")

    def _arrange(self, input: Tensor):
        """Arrange subsampled input before samples are taken for autograd sample computation.

        Subsampled_input may be arranged as desired before the autograd calculation of the
        gradient. After arrangement, the tensor should be the correct shape to be passed
        to _make_distribution. _rearrange should also be specified to return the final
        gradient to the original shape. Default does no arranging.

        Args:
            input: subsampled input

        Returns:
            Subsampled input correctly arranged to be passed to _make_distribution
        """
        return input

    def _rearrange(self, input: Tensor):
        """Rearrange sampled gradient after autograd gradient calculation.

        The opposite of _arrange, the gradient should be rearranged to match the
        original shape of subsampled_input. Default does no rearranging.

        Args:
            input: autograd-sampled gradient with _arrange-specified formatting

        Returns:
            Sampled gradients of shape [mc_samples, *subsampled_input.shape]
        """
        return input
