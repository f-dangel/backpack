from backpack.extensions.firstorder.batch_l2_grad.convtransposend import (
    BatchL2ConvTransposend,
)


class BatchL2ConvTranspose1d(BatchL2ConvTransposend):
    def __init__(self):
        super().__init__(N=1, params=["bias", "weight"])
