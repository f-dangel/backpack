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
    ) -> Tensor:
        self._checks(module)
        subsampled_input = Variable(
            subsample(module.input0, subsampling=subsampling), requires_grad=True
        )
        sampled_grads = self.compute_sampled_grads(subsampled_input, mc_samples) / sqrt(
            mc_samples
        )
        if module.reduction == "mean":
            sampled_grads /= sqrt(self._get_mean_normalization(module.input0))
        return sampled_grads

    def _checks(self, module):
        """Any checks to be performed. Default runs none."""
        return

    def compute_sampled_grads(self, subsampled_input, mc_samples):
        """Method to create the subsampled gradients.
        This can be overwritten by a faster method if a hand-crafted gradient is possible.
        Args:
            subsampled_input: input after subsampling
            mc_samples: number of samples
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
        """Create the sampling distribution for the negative log likelihood.
        Args:
            subsampled_input: input after subsampling
        Returns: torch.Distributions object
        """
        raise NotImplementedError

    @staticmethod
    def _get_mean_normalization(input: Tensor) -> int:
        raise NotImplementedError

    def hessian_is_psd(self) -> bool:
        raise NotImplementedError
