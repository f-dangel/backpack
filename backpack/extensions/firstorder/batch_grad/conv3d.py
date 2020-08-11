from backpack.core.derivatives.conv3d import Conv3DDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase


class BatchGradConv3d(BatchGradBase):
    def __init__(self):
        super().__init__(derivatives=Conv3DDerivatives(), params=["bias", "weight"])
