from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase


class BatchGradConvTranspose2d(BatchGradBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(), params=["bias", "weight"]
        )
