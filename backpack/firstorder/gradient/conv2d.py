import torch.nn
from ...core.layers import Conv2dConcat
from ...core.derivatives.conv2d import (Conv2DDerivatives,
                                        Conv2DConcatDerivatives)
from ...extensions import GRAD

from .base import GradBase


class GradConv2d(GradBase):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d,
            GRAD,
            Conv2DDerivatives(),
            params=["bias", "weight"])


class GradConv2dConcat(GradBase):
    def __init__(self):
        super().__init__(
            Conv2dConcat, GRAD, Conv2DConcatDerivatives(), params=["weight"])
