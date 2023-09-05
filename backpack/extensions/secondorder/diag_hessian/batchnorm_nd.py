"""DiagH extension for BatchNorm."""
from typing import Tuple, Union

from torch import Tensor, einsum, zeros, zeros_like
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule
from backpack.utils.errors import batch_norm_raise_error_if_train


class DiagHBatchNormNd(DiagHBaseModule):
    """DiagH and BatchDiagH extension for BatchNorm."""

    def __init__(self, sum_batch=True):
        """Initialization."""
        super().__init__(BatchNormNdDerivatives(), params=["bias", "weight"])
        self.sum_batch = sum_batch

    def check_hyperparameters_module_extension(
        self,
        ext: BackpropExtension,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
    ) -> None:  # noqa: D102
        batch_norm_raise_error_if_train(module)

    def extract_diagonal(
        self,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        S: Tensor,
        sum_batch: bool = True,
    ) -> Tensor:
        """Extract diagonal of ``(Jᵀ S) (Jᵀ S)ᵀ`` where ``J`` is the bias Jacobian.

        Args:
            module: BatchNorm layer for which the diagonal is extracted w.r.t. the bias.
            S: Backpropagated symmetric factorization of the loss Hessian. Has shape
                ``(V, *module.output.shape)``.
            sum_batch: Sum out the bias diagonal's batch dimension. Default: ``True``.

        Returns:
            Per-sample bias diagonal if ``sum_batch=False`` (shape
            ``(N, module.bias.shape)`` with batch size ``N``) or summed bias diagonal
            if ``sum_batch=True`` (shape ``module.bias.shape``).
        """
        # Pytorch repeats the identical bias terms in the last dimensions
        if module.input0.dim() > 3:
            JS = S.mean(dim=(3, -1))
        else:
            JS = S

        equation = f"vno->{'' if sum_batch else 'n'}o"

        return einsum(equation, JS**2)

    def bias(self, ext, module, g_inp, g_out, backproped):
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        if self.sum_batch is False:
            h_diag = zeros(
                module.input0.shape[0],
                *module.bias.shape,
                device=module.bias.device,
                dtype=module.bias.dtype,
            )
        else:
            h_diag = zeros_like(module.bias)

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(
                self.extract_diagonal(module, h_sqrt, sum_batch=self.sum_batch),
                alpha=sign,
            )

        return h_diag

    def weight(self, ext, module, g_inp, g_out, backproped):
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        if self.sum_batch is False:
            h_diag = zeros(
                module.input0.shape[0],
                *module.weight.shape,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
        else:
            h_diag = zeros_like(module.weight)

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(
                self.extract_diagonal(module, h_sqrt, sum_batch=self.sum_batch),
                alpha=sign,
            )

        return h_diag
