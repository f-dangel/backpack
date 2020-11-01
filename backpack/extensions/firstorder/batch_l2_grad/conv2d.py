from backpack.extensions.firstorder.batch_l2_grad.convnd import BatchL2Convnd


class BatchL2Conv2d(BatchL2Convnd):
    def __init__(self):
        super().__init__(N=2, params=["bias", "weight"])
