from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives

from .base import GradBaseModule


class GradBatchNorm1d(GradBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=BatchNormNdDerivatives(), params=["bias", "weight"]
        )
