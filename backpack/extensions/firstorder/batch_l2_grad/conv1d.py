from backpack.extensions.firstorder.batch_l2_grad.conv_base import BatchL2ConvBase


class BatchL2Conv1d(BatchL2ConvBase):
    def __init__(self):
        super().__init__(N=1, params=["bias", "weight"], convtranspose=False)
