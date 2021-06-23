from typing import Any

from torch import Tensor, einsum
from torch.nn import Linear

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives


class LinearDerivatives(BaseParameterDerivatives):
    """Partial derivatives for the Linear layer.

    Index conventions:
    ------------------
    * v: Free dimension
    * n: Batch dimension
    * o: Output dimension
    * i: Input dimension
    """

    def hessian_is_zero(self):
        return True

    def _jac_t_mat_prod(
        self, module: Linear, g_inp: Any, g_out: Any, mat: Tensor
    ) -> Tensor:
        """Batch-apply transposed Jacobian of the output w.r.t. the input.

        Args:
            module: Linear layer.
            g_inp: Gradients w.r.t. module input. Not required by the implementation.
            g_out: Gradients w.r.t. module output. Not required by the implementation.
            mat: Batch of ``V`` vectors of same shape as the layer output
                (``[N, *, out_features]``) to which the transposed output-input Jacobian
                is applied. Has shape ``[V, N, *, out_features]``.

        Returns:
            Batched transposed Jacobian vector products. Has shape
            ``[V, N, *, in_features]``.
        """
        return einsum("oi,vn...o->vn...i", module.weight.data, mat)

    def _jac_mat_prod(
        self, module: Linear, g_inp: Any, g_out: Any, mat: Tensor
    ) -> Tensor:
        """Batch-apply Jacobian of the output w.r.t. the input.

        Args:
            module: Linear layer.
            g_inp: Gradients w.r.t. module input. Not required by the implementation.
            g_out: Gradients w.r.t. module output. Not required by the implementation.
            mat: Batch of ``V`` vectors of same shape as the layer input
                (``[N, *, in_features]``) to which the output-input Jacobian is applied.
                Has shape ``[V, N, *, in_features]``.

        Returns:
            Batched Jacobian vector products. Has shape ``[V, N, *, out_features]``.
        """
        return einsum("oi,vn...i->vn...o", module.weight.data, mat)

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        jac = module.weight.data
        return einsum("ik,ij,jl->kl", (jac, mat, jac))

    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        """Apply Jacobian of the output w.r.t. the weight."""
        d_weight = module.input0
        return einsum("ni,voi->vno", (d_weight, mat))

    def _weight_jac_t_mat_prod(
        self, module: Linear, g_inp: Any, g_out: Any, mat: Tensor, sum_batch: int = True
    ) -> Tensor:
        """Batch-apply transposed Jacobian of the output w.r.t. the weight.

        Args:
            module: Linear layer.
            g_inp: Gradients w.r.t. module input. Not required by the implementation.
            g_out: Gradients w.r.t. module output. Not required by the implementation.
            mat: Batch of ``V`` vectors of same shape as the layer output
                (``[N, *, out_features]``) to which the transposed output-input Jacobian
                is applied. Has shape ``[V, N, *, out_features]``.
            sum_batch: Sum the result's batch axis. Default: ``True``.

        Returns:
            Batched transposed Jacobian vector products. Has shape
                ``[V, N, *module.weight.shape]`` when ``sum_batch`` is ``False``. With
                ``sum_batch=True``, has shape ``[V, *module.weight.shape]``.
        """
        d_weight = module.input0

        if self._has_additional_dims(module):
            # Flatten additional dimensions because they cannot be represented as
            # ellipsis. WAITING https://github.com/pytorch/pytorch/issues/45854
            d_weight = d_weight.flatten(start_dim=1, end_dim=-2)
            mat = mat.flatten(start_dim=2, end_dim=-2)
            equation = f"vnao,nai->v{'' if sum_batch else 'n'}oi"
        else:
            equation = f"vno,ni->v{'' if sum_batch else 'n'}oi"

        return einsum(equation, mat, d_weight)

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        """Apply Jacobian of the output w.r.t. the bias."""
        N = module.input0.size(0)
        return mat.unsqueeze(1).expand(-1, N, -1)

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Apply transposed Jacobian of the output w.r.t. the bias."""
        if sum_batch:
            N_axis = 1
            return mat.sum(N_axis)
        else:
            return mat

    @staticmethod
    def _has_additional_dims(module: Linear) -> bool:
        """Return whether the input to a linear layer has additional (>1) dimensions.

        The input to a linear layer may have shape ``[N, *, out_features]``.
        It has additional dimensions if ``*`` is non-empty.

        Args:
            module: Linear layer.

        Returns:
            Whether the input has hidden dimensions.
        """
        return module.input0.dim() != 2
