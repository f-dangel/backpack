import torch.nn
from ...core.layers import LinearConcat
from ...core.derivatives.linear import (LinearDerivatives,
                                        LinearConcatDerivatives)
from ...extensions import GRAD

from .base import GradBase


class GradLinear(GradBase):
    def __init__(self):
        super().__init__(
            torch.nn.Linear,
            GRAD,
            LinearDerivatives(),
            params=["bias", "weight"])


class GradLinearConcat(GradBase):
    def __init__(self):
        super().__init__(
            LinearConcat, GRAD, LinearConcatDerivatives(), params=["weight"])
