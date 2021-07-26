"""Contains batch_l2 extension for BatchNorm."""
from typing import Tuple, Union

from torch import Tensor
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.firstorder.batch_l2_grad.batch_l2_base import BatchL2Base
from backpack.utils.errors import batch_norm_raise_error_if_train


class BatchL2BatchNorm(BatchL2Base):
    """batch_l2 extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(["weight", "bias"], BatchNormNdDerivatives())

    def check_hyperparameters_module_extension(
        self,
        ext: BackpropExtension,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
    ) -> None:  # noqa: D102
        batch_norm_raise_error_if_train(module)
