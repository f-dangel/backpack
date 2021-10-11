"""Contains partial derivatives for the ``torch.nn.Linear`` layer."""
from typing import List, Tuple

from torch import Size, Tensor, einsum
from torch.nn import Linear

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils.subsampling import subsample


class LinearDerivatives(BaseParameterDerivatives):
    """Partial derivatives for the Linear layer.

    Index conventions:
    ------------------
    * v: Free dimension
    * n: Batch dimension
    * o: Output dimension
    * i: Input dimension
    """

    def hessian_is_zero(self, module: Linear) -> bool:
        """Linear layer output is linear w.r.t. to its input.

        Args:
            module: current module

        Returns:
            True
        """
        return True

    def _jac_t_mat_prod(
        self,
        module: Linear,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Batch-apply transposed Jacobian of the output w.r.t. the input.

        Args:
            module: Linear layer.
            g_inp: Gradients w.r.t. module input. Not required by the implementation.
            g_out: Gradients w.r.t. module output. Not required by the implementation.
            mat: Batch of ``V`` vectors of same shape as the layer output
                (``[N, *, out_features]``) to which the transposed output-input Jacobian
                is applied. Has shape ``[V, N, *, out_features]``; but if used with
                sub-sampling, ``N`` is replaced by ``len(subsampling)``.
            subsampling: Indices of active samples. ``None`` means all samples.

        Returns:
            Batched transposed Jacobian vector products. Has shape
            ``[V, N, *, in_features]``. If used with sub-sampling, ``N`` is replaced
            by ``len(subsampling)``.
        """
        return einsum("vn...o,oi->vn...i", mat, module.weight)

    def _jac_mat_prod(
        self, module: Linear, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
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
        return einsum("oi,vn...i->vn...o", module.weight, mat)

    def ea_jac_t_mat_jac_prod(
        self, module: Linear, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        """Expectation approximation of outer product with input-output Jacobian.

        Used for KFRA backpropagation: ``mat ← E(Jₙᵀ mat Jₙ) = 1/N ∑ₙ Jₙᵀ mat Jₙ``.

        Args:
            module: Linear layer.
            g_inp: Gradients w.r.t. module input. Not required by the implementation.
            g_out: Gradients w.r.t. module output. Not required by the implementation.
            mat: Matrix of shape
                ``[module.output.numel() // N, module.output.numel() // N]``.

        Returns:
            Matrix of shape
            ``[module.input0.numel() // N, module.input0.numel() // N]``.
        """
        add_features = self._get_additional_dims(module).numel()
        in_features, out_features = module.in_features, module.out_features

        result = mat.reshape(add_features, out_features, add_features, out_features)
        result = einsum("ik,xiyj,jl->xkyl", module.weight, result, module.weight)

        return result.reshape(in_features * add_features, in_features * add_features)

    def _weight_jac_mat_prod(
        self, module: Linear, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        """Batch-apply Jacobian of the output w.r.t. the weight.

        Args:
            module: Linear layer.
            g_inp: Gradients w.r.t. module input. Not required by the implementation.
            g_out: Gradients w.r.t. module output. Not required by the implementation.
            mat: Batch of ``V`` vectors of shape ``module.weight.shape`` to which the
                transposed output-input Jacobian is applied. Has shape
                ``[V, *module.weight.shape]``.

        Returns:
            Batched Jacobian vector products. Has shape
            ``[V, N, *module.output.shape]``.
        """
        return einsum("n...i,voi->vn...o", module.input0, mat)

    def _weight_jac_t_mat_prod(
        self,
        module: Linear,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: int = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Batch-apply transposed Jacobian of the output w.r.t. the weight.

        Args:
            module: Linear layer.
            g_inp: Gradients w.r.t. module input. Not required by the implementation.
            g_out: Gradients w.r.t. module output. Not required by the implementation.
            mat: Batch of ``V`` vectors of same shape as the layer output
                (``[N, *, out_features]``) to which the transposed output-input Jacobian
                is applied. Has shape ``[V, N, *, out_features]`` if subsampling is not
                used, otherwise ``N`` must be ``len(subsampling)`` instead.
            sum_batch: Sum the result's batch axis. Default: ``True``.
            subsampling: Indices of samples along the output's batch dimension that
                should be considered. Defaults to ``None`` (use all samples).

        Returns:
            Batched transposed Jacobian vector products. Has shape
            ``[V, N, *module.weight.shape]`` when ``sum_batch`` is ``False``. With
            ``sum_batch=True``, has shape ``[V, *module.weight.shape]``. If sub-
            sampling is used, ``N`` must be ``len(subsampling)`` instead.
        """
        d_weight = subsample(module.input0, subsampling=subsampling)

        equation = f"vn...o,n...i->v{'' if sum_batch else 'n'}oi"
        return einsum(equation, mat, d_weight)

    def _bias_jac_mat_prod(
        self, module: Linear, g_inp: Tuple[Tensor], g_out: Tuple[Tensor], mat: Tensor
    ) -> Tensor:
        """Batch-apply Jacobian of the output w.r.t. the bias.

        Args:
            module: Linear layer.
            g_inp: Gradients w.r.t. module input. Not required by the implementation.
            g_out: Gradients w.r.t. module output. Not required by the implementation.
            mat: Batch of ``V`` vectors of shape ``module.bias.shape`` to which the
                transposed output-input Jacobian is applied. Has shape
                ``[V, *module.bias.shape]``.

        Returns:
            Batched Jacobian vector products. Has shape
            ``[V, N, *module.output.shape]``.
        """
        N = module.input0.shape[0]
        additional_dims = list(self._get_additional_dims(module))

        for _ in range(len(additional_dims) + 1):
            mat = mat.unsqueeze(1)

        expand = [-1, N] + additional_dims + [-1]

        return mat.expand(*expand)

    def _bias_jac_t_mat_prod(
        self,
        module: Linear,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: int = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Batch-apply transposed Jacobian of the output w.r.t. the bias.

        Args:
            module: Linear layer.
            g_inp: Gradients w.r.t. module input. Not required by the implementation.
            g_out: Gradients w.r.t. module output. Not required by the implementation.
            mat: Batch of ``V`` vectors of same shape as the layer output
                (``[N, *, out_features]``) to which the transposed output-input Jacobian
                is applied. Has shape ``[V, N, *, out_features]``.
            sum_batch: Sum the result's batch axis. Default: ``True``.
            subsampling: Indices of samples along the output's batch dimension that
                should be considered. Defaults to ``None`` (use all samples).

        Returns:
            Batched transposed Jacobian vector products. Has shape
            ``[V, N, *module.bias.shape]`` when ``sum_batch`` is ``False``. With
            ``sum_batch=True``, has shape ``[V, *module.bias.shape]``. If sub-
            sampling is used, ``N`` is replaced by ``len(subsampling)``.
        """
        equation = f"vn...o->v{'' if sum_batch else 'n'}o"
        return einsum(equation, mat)

    @staticmethod
    def _get_additional_dims(module: Linear) -> Size:
        """Return the shape of additional dimensions in the input to a linear layer.

        Args:
            module: A linear layer.

        Returns:
            Shape of the additional dimensions. Corresponds to ``*`` in the
            input shape ``[N, *, out_features]``.
        """
        return module.input0.shape[1:-1]
