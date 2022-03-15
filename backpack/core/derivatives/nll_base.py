"""Partial derivative bases for NLL losses."""
from math import sqrt
from typing import List, Tuple

from torch import Size, Tensor, reshape
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
        samples = reshape(
            samples, (mc_samples, len(subsampled_input), len(subsampled_input[0]))
        )
        samples = self._sqrt(samples) / sqrt(mc_samples)

        if module.reduction == "mean":
            samples = self._mean_reduction(samples, module.input0)

        return self._post_process(samples)

    def _checks(self, module):
        """
        Default runs no checks.
        """
        return

    def _make_distribution(self, subsampled_input, mc_samples):
        raise NotImplementedError

    def _sqrt(self, samples):
        return samples

    def _post_process(self, samples):
        return samples

    def _mean_reduction(self, samples, input0):
        return samples / sqrt(input0.numel())

    def hessian_is_psd(self) -> bool:
        raise NotImplementedError
