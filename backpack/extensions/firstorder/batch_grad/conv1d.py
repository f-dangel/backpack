from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase


class BatchGradConv1d(BatchGradBase):
    def __init__(self):
        super().__init__(derivatives=Conv1DDerivatives(), params=["bias", "weight"])
