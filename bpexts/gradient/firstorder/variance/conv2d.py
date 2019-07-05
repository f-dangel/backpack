import torch.nn
from ...extensions import VARIANCE
from ..firstorder import FirstOrderExtension
from ..sumgradsquared.conv2d import SGSConv2d
from ..gradient.conv2d import GradConv2d
from .base import variance_from


class VarianceConv2d(FirstOrderExtension):

    def __init__(self):
        super().__init__(torch.nn.Conv2d, VARIANCE, params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        N = grad_output[0].shape[0]
        return variance_from(
            GradConv2d().bias(module, grad_input, grad_output),
            SGSConv2d().bias(module, grad_input, grad_output),
            N
        )

    def weight(self, module, grad_input, grad_output):
        N = grad_output[0].shape[0]
        return variance_from(
            GradConv2d().weight(module, grad_input, grad_output),
            SGSConv2d().weight(module, grad_input, grad_output),
            N
        )


EXTENSIONS = [VarianceConv2d()]
