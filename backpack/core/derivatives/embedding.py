"""Derivatives for Embedding."""
from typing import List, Tuple

from torch import Tensor, einsum, zeros
from torch.nn import Embedding

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils.subsampling import subsample


class EmbeddingDerivatives(BaseParameterDerivatives):
    """Derivatives for Embedding.

    Note:
        These derivatives assume the batch axis to be at position 0.

    Index convention:
    v - free axis
    n - batch axis
    s - num_embeddings
    h - embedding_dim
    """

    def _jac_t_mat_prod(
        self,
        module: Embedding,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        raise NotImplementedError(
            "Derivative w.r.t. input not defined: Input to Embedding has type long."
            " But only float and complex dtypes can require gradients in PyTorch."
        )

    def _weight_jac_t_mat_prod(
        self,
        module: Embedding,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        self._check_parameters(module)

        input0 = subsample(module.input0, subsampling=subsampling)
        delta = zeros(module.num_embeddings, *input0.shape, device=mat.device)
        for s in range(module.num_embeddings):
            delta[s] = input0 == s
        equation = f"sn...,vn...h->v{'' if sum_batch else 'n'}sh"
        return einsum(equation, delta, mat)

    def _check_parameters(self, module: Embedding) -> None:
        if module.padding_idx is not None:
            raise NotImplementedError("Only padding_idx=None supported.")
        elif module.max_norm is not None:
            raise NotImplementedError("Only max_norm=None supported.")
        elif module.scale_grad_by_freq:
            raise NotImplementedError("Only scale_grad_by_freq=False supported.")
        elif module.sparse:
            raise NotImplementedError("Only sparse=False supported.")

    def hessian_is_zero(self, module: Embedding) -> bool:  # noqa: D102
        return False
