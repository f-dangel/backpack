import torch.nn
from ...extensions import BATCH_GRAD
from ...core.layers import LinearConcat

from ...core.derivatives.linear import (LinearDerivatives,
                                        LinearConcatDerivatives)

from .base import BatchGradBase


class BatchGradLinear(BatchGradBase):
    def __init__(self):
        super().__init__(
            torch.nn.Linear,
            BATCH_GRAD,
            LinearDerivatives(),
            params=["bias", "weight"])


class BatchGradLinearConcat(BatchGradBase):
    def __init__(self):
        super().__init__(
            LinearConcat,
            BATCH_GRAD,
            LinearConcatDerivatives(),
            params=["weight"])


EXTENSIONS = [BatchGradLinear(), BatchGradLinearConcat()]
