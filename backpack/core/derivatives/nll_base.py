"""Partial derivative bases for NLL losses."""
from math import sqrt
from typing import List, Tuple

from torch import Size, Tensor, enable_grad, stack
from torch.autograd import Variable, grad
from torch.nn import CrossEntropyLoss

from backpack.core.derivatives.basederivatives import BaseLossDerivatives
from backpack.utils.subsampling import subsample


class NLLLossDerivatives(BaseLossDerivatives):
    """Partial derivative bases for NLL loss."""

    def _sqrt_hessian_sampled(
        self,
        module: CrossEntropyLoss,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mc_samples: int = 1,
        subsampling: List[int] = None,
        use_dist: bool = False,
    ) -> Tensor:
        """Method to approximate the square root Hessian through Monte Carlo sampling.
        For a given loss, either _make_distribution or compute_sampled_grads must be
        implemented.
        Optionally, it is possible to specify _checks to perform before calculating
        the loss gradient.
        For use in mean mode, _get_mean_normalization must be implemented.
        Args:
            module:
            g_inp:
            g_out:
            mc_samples: number of Monte Carlo samples to take
            subsampling: Indices of samples that are sliced along the dimension.
            Defaults to ``None`` (use all samples).
        Returns:
            Monte Carlo sampled gradients
        """
        self._checks(module)
        subsampled_input = Variable(
            subsample(module.input0, subsampling=subsampling), requires_grad=True
        )
        sampled_grads = self.compute_sampled_grads(
            subsampled_input, mc_samples, use_dist
        ) / sqrt(mc_samples)
        if module.reduction == "mean":
            sampled_grads /= sqrt(self._get_mean_normalization(module.input0))
        return sampled_grads

    def _checks(self, module):
        """Any checks to be performed. Default runs none."""
        return

    def compute_sampled_grads(
        self, subsampled_input, mc_samples, use_dist: bool = False
    ):
        """Method to create the sampled gradients.
        For loss functions that can be expressed as a Negative Log Likelihood (NLL), the
        Maximum a-Posteriori (MAP) estimate of the network parameters can be written as
        𝜃ₘₐₚ = argmin_𝜃 (𝑟(𝜃)+𝑙(xₙ,yₙ;𝜃)) where 𝑟 is the regularizer and 𝑙 is the loss
        function, defined here as 𝑙(xₙ,yₙ;𝜃)= −log p(yₙ | f_𝜃(xₙ)).
        compute_sampled_grads calculates and returns the gradient of the loss with respect
        to each of the randomly drawn samples. To use this function, the user must implement
        the functions _make_distribution.
        Alternatively, if a faster gradient calculation is known for the loss function, the
        user may overwrite this method entirely with a hand-crafted method. Then the method
        should return the gradient of the loss with respect to the subsampled input for each
        of the Monte Carlo samples
        Args:
            subsampled_input: input after subsampling
            mc_samples: number of samples
            use_dist: boolean to use NLL version of compute_sampled_grads for testing
        Returns:
            sampled gradient of shape [mc_samples, *subsampled_input.shape]
        """
        with enable_grad():
            gradient = []
            dist = self._make_distribution(subsampled_input)
            y_tilde = dist.sample(sample_shape=Size([mc_samples]))
            loss_tilde = -dist.log_prob(y_tilde)
            for m in range(mc_samples):
                gradient.append(
                    grad(
                        outputs=loss_tilde[m].sum(),
                        inputs=subsampled_input,
                        retain_graph=True,
                    )[0]
                )
        return stack(gradient)

    def _make_distribution(self, subsampled_input):
        """Create the sampling distribution for the NLL.
        This should be in the form of a torch.Distributions object, such that
        the desired loss 𝑙(xₙ,yₙ;𝜃)= −log p(yₙ | f_𝜃(xₙ)).
        Args:
            subsampled_input: input after subsampling
        Returns:
            torch.Distributions object
        """
        raise NotImplementedError

    @staticmethod
    def _get_mean_normalization(input: Tensor) -> int:
        """Normalization factor for mean mode.
        If used in mean mode, the normalization factor must be provided.
        Args:
            input: module input (before subsampling)
        Returns:
            normalization factor for mean mode
        """
        raise NotImplementedError
