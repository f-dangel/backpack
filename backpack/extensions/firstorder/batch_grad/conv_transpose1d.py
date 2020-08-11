from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase


class BatchGradConvTranspose1d(BatchGradBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(), params=["bias", "weight"]
        )
