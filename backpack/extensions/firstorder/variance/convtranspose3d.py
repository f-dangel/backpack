from backpack.extensions.firstorder.gradient.convtranspose3d import GradConvTranspose3d
from backpack.extensions.firstorder.sum_grad_squared.convtranspose3d import (
    SGSConvTranspose3d,
)

from .variance_base import VarianceBaseModule


class VarianceConvTranspose3d(VarianceBaseModule):
    def __init__(self):
        super().__init__(
            params=["bias", "weight"],
            grad_extension=GradConvTranspose3d(),
            sgs_extension=SGSConvTranspose3d(),
        )
