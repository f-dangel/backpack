from backpack.extensions.firstorder.batch_l2_grad.convnd import BatchL2ConvND


class BatchL2Conv2d(BatchL2ConvND):
    def __init__(self):
        super().__init__(N=2, params=["bias", "weight"])
