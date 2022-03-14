"""Partial derivative bases for NLL losses."""
from math import sqrt
from typing import Callable, Dict, List, Tuple

from einops import rearrange
from torch import Tensor, diag, diag_embed, einsum, eye, multinomial, ones_like, softmax
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot

from backpack.core.derivatives.basederivatives import BaseLossDerivatives
from backpack.utils.subsampling import subsample


class NLLLossDerivatives(BaseLossDerivatives):
    """Partial derivative bases for NLL loss.
    """

    def _sqrt_hessian(
        self,
        module: CrossEntropyLoss,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:
        # TODO
        raise NotImplementedError

    def _sqrt_hessian_sampled(
        self,
        module: CrossEntropyLoss,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mc_samples: int = 1,
        subsampling: List[int] = None,
    ) -> Tensor:
        self._checks(module)
        M = mc_samples
        N, D = module.input0.shape
        dist, to_sample = self._make_distribution(module, subsampling, N, M, D)
        samples = dist.sample(to_sample)
        raise NotImplementedError

    def _sum_hessian(
        self, module: CrossEntropyLoss, g_inp: Tuple[Tensor], g_out: Tuple[Tensor]
    ) -> Tensor:
        # TODO
        raise NotImplementedError

    def _make_hessian_mat_prod(
        self, module: CrossEntropyLoss, g_inp: Tuple[Tensor], g_out: Tuple[Tensor]
    ) -> Callable[[Tensor], Tensor]:
        # TODO
        raise NotImplementedError

    def hessian_is_psd(self) -> bool:
        """Return whether loss Hessian is positive semi-definite.
        """
        raise NotImplementedError

    def _checks(self, module):
        """
        Default runs no checks
        """
        return

    def _make_distribution(self, module, subsampling, N, M, D):
        raise NotImplementedError
