"""DiagH extension for BatchNorm."""
from typing import Tuple, Union

from torch import Tensor
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule
from backpack.utils.errors import batch_norm_raise_error_if_train


class DiagHBatchNormNd(DiagHBaseModule):
    """DiagH and BatchDiagH extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(BatchNormNdDerivatives(), params=None)

    def check_hyperparameters_module_extension(
        self,
        ext: BackpropExtension,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
    ) -> None:  # noqa: D102
        batch_norm_raise_error_if_train(module)
