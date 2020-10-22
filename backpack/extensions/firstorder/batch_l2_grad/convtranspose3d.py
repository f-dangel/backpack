from backpack.extensions.firstorder.batch_l2_grad.conv_base import BatchL2ConvBase


class BatchL2ConvTranspose3d(BatchL2ConvBase):
    def __init__(self):
        super().__init__(N=3, params=["bias", "weight"], convtranspose=True)
