from backpack.extensions.firstorder.batch_l2_grad.convtransposend import (
    BatchL2ConvTransposeND,
)


class BatchL2ConvTranspose3d(BatchL2ConvTransposeND):
    def __init__(self):
        super().__init__(N=3, params=["bias", "weight"])
