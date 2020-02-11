from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase


class BatchGradLinear(BatchGradBase):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])
