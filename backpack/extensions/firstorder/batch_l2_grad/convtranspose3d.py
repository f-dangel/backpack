from backpack.extensions.firstorder.batch_l2_grad.convtransposend import (
    BatchL2ConvTransposend,
)


class BatchL2ConvTranspose3d(BatchL2ConvTransposend):
    def __init__(self):
        super().__init__(N=3, params=["bias", "weight"])
