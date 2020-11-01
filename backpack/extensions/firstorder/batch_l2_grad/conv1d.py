from backpack.extensions.firstorder.batch_l2_grad.convnd import BatchL2Convnd


class BatchL2Conv1d(BatchL2Convnd):
    def __init__(self):
        super().__init__(N=1, params=["bias", "weight"])
