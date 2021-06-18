from backpack.core.derivatives.batchnorm_nd import BatchNorm1dDerivatives

from .base import GradBaseModule


class GradBatchNorm1d(GradBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=BatchNorm1dDerivatives(), params=["bias", "weight"]
        )
