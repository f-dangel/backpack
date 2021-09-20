"""Variance extension for BatchNorm."""
from typing import Tuple, Union

from torch import Tensor
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.firstorder.gradient.batchnorm_nd import GradBatchNormNd
from backpack.extensions.firstorder.sum_grad_squared.batchnorm_nd import SGSBatchNormNd
from backpack.extensions.firstorder.variance.variance_base import VarianceBaseModule
from backpack.utils.errors import batch_norm_raise_error_if_train


class VarianceBatchNormNd(VarianceBaseModule):
    """Variance extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(["weight", "bias"], GradBatchNormNd(), SGSBatchNormNd())

    def check_hyperparameters_module_extension(
        self,
        ext: BackpropExtension,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
    ) -> None:  # noqa: D102
        batch_norm_raise_error_if_train(module)
