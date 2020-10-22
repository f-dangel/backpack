from backpack.extensions.firstorder.batch_l2_grad.conv_base import BatchL2ConvBase


class BatchL2Conv2d(BatchL2ConvBase):
    def __init__(self):
        super().__init__(N=2, params=["bias", "weight"], convtranspose=False)
