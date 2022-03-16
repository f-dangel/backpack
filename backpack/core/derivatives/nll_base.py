"""Partial derivative bases for NLL losses."""
from math import sqrt
from typing import List, Tuple

from torch import Size, Tensor
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
    ) -> Tensor:
        self._checks(module)

        subsampled_input = subsample(module.input0, subsampling=subsampling)
        dist = self._make_distribution(subsampled_input, mc_samples)

        samples = dist.sample(sample_shape=Size([mc_samples]))
        samples = self._sqrt(samples) / sqrt(mc_samples)

        if module.reduction == "mean":
            samples = self._mean_reduction(samples, module.input0)

        return self._post_process(samples)

    def _checks(self, module):
        """
        Any checks to be performed.
        """
        raise NotImplementedError

    def _make_distribution(self, subsampled_input, mc_samples):
        """
        Create the sampling distribution for the negative log likelihood.
        Args:
            subsampled_input: input after subsampling
            mc_samples: number of samples
        Returns: torch.Distributions object
        """
        raise NotImplementedError

    def _sqrt(self, samples):
        """
        Adjust the samples to get the correct Hessian. Default does nothing.
        Args:
            samples: samples taken
        Returns: corrected samples.
        """
        return samples

    def _post_process(self, samples):
        """
        Any final processing.
        Args:
            samples: sampled Hessian before final processing.
        Returns: final Hessian
        """
        raise NotImplementedError

    def _mean_reduction(self, samples, input0):
        """
        Take the mean. Default takes this over total elements.
        Args:
            samples: square root Hessian
            input0: original input
        Returns: mean Hessian
        """
        return samples / sqrt(input0.numel())

    def hessian_is_psd(self) -> bool:
        raise NotImplementedError
