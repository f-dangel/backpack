import torch.nn
from ...gradient.linear import GradLinear
from ...extensions import VARIANCE
from ..firstorder import FirstOrderExtension
from ..sumgradsquared.linear import SGSLinear
from .base import variance_from


class VarianceLinear(FirstOrderExtension):

    def __init__(self):
        super().__init__(torch.nn.Linear, VARIANCE, params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        N = grad_output[0].shape[0]
        return variance_from(
            GradLinear().bias(module, grad_input, grad_output),
            SGSLinear().bias(module, grad_input, grad_output),
            N
        )

    def weight(self, module, grad_input, grad_output):
        N = grad_output[0].shape[0]
        return variance_from(
            GradLinear().weight(module, grad_input, grad_output),
            SGSLinear().weight(module, grad_input, grad_output),
            N
        )


EXTENSIONS = [VarianceLinear()]
