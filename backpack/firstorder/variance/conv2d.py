import torch.nn
from ...core.layers import Conv2dConcat
from ...extensions import VARIANCE
from ..firstorder import FirstOrderExtension
from ..sumgradsquared.conv2d import SGSConv2d, SGSConv2dConcat
from ..gradient.conv2d import GradConv2d, GradConv2dConcat
from .base import variance_from


class VarianceConv2d(FirstOrderExtension):
    def __init__(self):
        super().__init__(torch.nn.Conv2d, VARIANCE, params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        N = grad_output[0].shape[0]
        return variance_from(
            GradConv2d().bias(module, grad_input, grad_output),
            SGSConv2d().bias(module, grad_input, grad_output), N)

    def weight(self, module, grad_input, grad_output):
        N = grad_output[0].shape[0]
        return variance_from(
            GradConv2d().weight(module, grad_input, grad_output),
            SGSConv2d().weight(module, grad_input, grad_output), N)


class VarianceConv2dConcat(FirstOrderExtension):
    def __init__(self):
        super().__init__(Conv2dConcat, VARIANCE, params=["weight"])

    def weight(self, module, grad_input, grad_output):
        N = grad_output[0].shape[0]
        return variance_from(
            GradConv2dConcat().weight(module, grad_input, grad_output),
            SGSConv2dConcat().weight(module, grad_input, grad_output), N)


EXTENSIONS = [VarianceConv2d(), VarianceConv2dConcat()]
