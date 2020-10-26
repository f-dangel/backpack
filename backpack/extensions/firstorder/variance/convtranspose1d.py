from backpack.extensions.firstorder.gradient.convtranspose1d import GradConvTranspose1d
from backpack.extensions.firstorder.sum_grad_squared.convtranspose1d import (
    SGSConvTranspose1d,
)

from .variance_base import VarianceBaseModule


class VarianceConvTranspose1d(VarianceBaseModule):
    def __init__(self):
        super().__init__(
            params=["bias", "weight"],
            grad_extension=GradConvTranspose1d(),
            sgs_extension=SGSConvTranspose1d(),
        )
