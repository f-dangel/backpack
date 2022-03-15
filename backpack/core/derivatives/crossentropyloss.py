"""NLL extention for Cross-Entropy Loss."""
from abc import ABC
from math import sqrt
from typing import Dict, Tuple

from einops import rearrange
from torch import Tensor, softmax
from torch.distributions import OneHotCategorical
from torch.nn import CrossEntropyLoss

from backpack.core.derivatives.nll_base import NLLLossDerivatives


class CrossEntropyLossDerivatives(NLLLossDerivatives, ABC):
    """Partial derivatives for cross-entropy loss.

    This comes from the one-hot encoded Categorical distribution.
    """
    def _checks(self, module):
        self._check_2nd_order_parameters(module)

    def _make_distribution(self, subsampled_input, mc_samples):
        probs = softmax(subsampled_input, dim=1)
        probs, *rearrange_info = self._merge_batch_and_additional(probs)
        self.rearrange_info = rearrange_info
        self.probs_unsqeezed = probs.unsqueeze(0).repeat(mc_samples, 1, 1)
        return OneHotCategorical(probs)

    def _sqrt(self, samples):
        return self.probs_unsqeezed - samples

    def _mean_reduction(self, samples, input0):
        return samples / sqrt(input0.numel() // input0.shape[1])

    def _post_process(self, samples):
        return self._ungroup_batch_and_additional(samples, *self.rearrange_info)

    def _check_2nd_order_parameters(self, module: CrossEntropyLoss) -> None:
        """Verify that the parameters are supported by 2nd-order quantities.

        Args:
            module: Extended CrossEntropyLoss module

        Raises:
            NotImplementedError: If module's setting is not implemented.
        """
        implemented_ignore_index = -100
        implemented_weight = None

        if module.ignore_index != implemented_ignore_index:
            raise NotImplementedError(
                "Only default ignore_index ({}) is implemented, got {}".format(
                    implemented_ignore_index, module.ignore_index
                )
            )

        if module.weight != implemented_weight:
            raise NotImplementedError(
                "Only default weight ({}) is implemented, got {}".format(
                    implemented_weight, module.weight
                )
            )

    @staticmethod
    def _merge_batch_and_additional(
        probs: Tensor,
    ) -> Tuple[Tensor, str, Dict[str, int]]:
        """Rearranges the input if it has additional axes.

        Treat additional axes like batch axis, i.e. group ``n c d1 d2 -> (n d1 d2) c``.

        Args:
            probs: the tensor to rearrange

        Returns:
            a tuple containing
                - probs: the rearranged tensor
                - str_d_dims: a string representation of the additional dimensions
                - d_info: a dictionary encoding the size of the additional dimensions
        """
        leading = 2
        additional = probs.dim() - leading

        str_d_dims: str = "".join(f"d{i} " for i in range(additional))
        d_info: Dict[str, int] = {
            f"d{i}": probs.shape[leading + i] for i in range(additional)
        }

        probs = rearrange(probs, f"n c {str_d_dims} -> (n {str_d_dims}) c")

        return probs, str_d_dims, d_info

    @staticmethod
    def _ungroup_batch_and_additional(
        tensor: Tensor, str_d_dims, d_info, free_axis: int = 1
    ) -> Tensor:
        """Rearranges output if it has additional axes.

        Used with group_batch_and_additional.

        Undoes treating additional axes like batch axis and assumes an number of
        additional free axes (``v``) were added, i.e. un-groups
        ``v (n d1 d2) c -> v n c d1 d2``.

        Args:
            tensor: the tensor to rearrange
            str_d_dims: a string representation of the additional dimensions
            d_info: a dictionary encoding the size of the additional dimensions
            free_axis: Number of free leading axes. Default: ``1``.

        Returns:
            the rearranged tensor

        Raises:
            NotImplementedError: If ``free_axis != 1``.
        """
        if free_axis != 1:
            raise NotImplementedError(f"Only supports free_axis=1. Got {free_axis}.")

        return rearrange(
            tensor, f"v (n {str_d_dims}) c -> v n c {str_d_dims}", **d_info
        )

    def hessian_is_psd(self) -> bool:
        return True
