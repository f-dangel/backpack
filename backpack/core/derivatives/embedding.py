"""Derivatives for Embedding."""
from typing import List, Tuple

from torch import Tensor, einsum, flatten, zeros
from torch.nn import Embedding

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils import TORCH_VERSION_AT_LEAST_1_9_0
from backpack.utils.subsampling import subsample


class EmbeddingDerivatives(BaseParameterDerivatives):
    """Derivatives for Embedding.

    These derivatives assume that the batch axis is at position 0.
    This might be different if used in combination with an RNN module that has
    batch_first=False. However, this mode is not supported in BackPACK.

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
            "The input tensor to Embedding is of type long. However, in PyTorch, only "
            "Tensors of floating point and complex dtype can require gradients. "
            "Therefore, a derivative wrt input is not defined."
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
        if TORCH_VERSION_AT_LEAST_1_9_0:
            equation = f"sn...,vn...h->v{'' if sum_batch else 'n'}sh"
        elif delta.dim() >= 3:
            equation = f"snx,vnxh->v{'' if sum_batch else 'n'}sh"
            delta = flatten(delta, start_dim=2, end_dim=-1)
            mat = flatten(mat, start_dim=2, end_dim=-2)
        else:
            equation = f"sn,vnh->v{'' if sum_batch else 'n'}sh"
        return einsum(equation, delta, mat)

    def _check_parameters(self, module: Embedding) -> None:
        if module.padding_idx is not None:
            raise NotImplementedError("Only padding_idx=None supported.")
        elif module.max_norm is not None:
            raise NotImplementedError("Only max_norm=None supported.")
        elif module.scale_grad_by_freq:
            raise NotImplementedError("Only scale_grad_by_freq=False supported.")
        elif module.sparse:  # TODO sparse might be supported -> test
            raise NotImplementedError("Only sparse=False supported.")

    def hessian_is_zero(self, module: Embedding) -> bool:  # noqa: D102
        return False  # TODO discuss
