"""NLL extention for Cross-Entropy Loss."""
from abc import ABC
from backpack.core.derivatives.nll_base import NLLLossDerivatives
from torch.nn import CrossEntropyLoss
from torch.distributions import OneHotCategorical
from backpack.utils.subsampling import subsample
from typing import List, Tuple, Dict
from torch import softmax, Tensor
from einops import rearrange


class CrossEntropyLossDerivatives(NLLLossDerivatives, ABC):
    def hessian_is_psd(self) -> bool:
        """Return whether loss Hessian is positive semi-definite.

        Returns:
            True
        """
        return True

    def _checks(self, module):
        self._check_2nd_order_parameters(module)

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

    def _make_distribution(self, module, subsampling, M, N, D):
        probs = self._get_probs(module, subsampling=subsampling)
        probs, *rearrange_info = self._merge_batch_and_additional(probs)
        return OneHotCategorical(probs)

    @staticmethod
    def _get_probs(module: CrossEntropyLoss, subsampling: List[int] = None) -> Tensor:
        """Compute the softmax probabilities from the module input.

        Args:
            module: cross-entropy loss with I/O.
            subsampling: Indices of samples to be considered. Default of ``None`` uses
                the full mini-batch.

        Returns:
            Softmax probabilites
        """
        input0 = subsample(module.input0, subsampling=subsampling)
        return softmax(input0, dim=1)

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