import torch.nn
from ...core.layers import Conv2dConcat
from ...extensions import BATCH_GRAD
from ...core.derivatives.conv2d import (Conv2DDerivatives,
                                        Conv2DConcatDerivatives)

from .base import BatchGradBase


class BatchGradConv2d(BatchGradBase):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d,
            BATCH_GRAD,
            Conv2DDerivatives(),
            params=["bias", "weight"])


class BatchGradConv2dConcat(BatchGradBase):
    def __init__(self):
        super().__init__(
            Conv2dConcat,
            BATCH_GRAD,
            Conv2DConcatDerivatives(),
            params=["weight"])


EXTENSIONS = [BatchGradConv2d(), BatchGradConv2dConcat()]
