from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sum_grad_base import SumGradBase


class SGSConv1d(SumGradBase):
    def __init__(self):
        super().__init__(derivatives=Conv1DDerivatives, params=["bias", "weight"])
