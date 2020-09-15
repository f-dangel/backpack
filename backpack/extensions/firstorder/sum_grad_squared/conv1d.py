from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sgs_base import SGSBase


class SGSConv1d(SGSBase):
    def __init__(self):
        super().__init__(derivatives=Conv1DDerivatives(), params=["bias", "weight"])
