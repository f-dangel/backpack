from backpack.extensions.firstorder.gradient.convtranspose2d import GradConvTranspose2d
from backpack.extensions.firstorder.sum_grad_squared.convtranspose2d import (
    SGSConvTranspose2d,
)

from .variance_base import VarianceBaseModule


class VarianceConvTranspose2d(VarianceBaseModule):
    def __init__(self):
        super().__init__(
            params=["bias", "weight"],
            grad_extension=GradConvTranspose2d(),
            sgs_extension=SGSConvTranspose2d(),
        )
