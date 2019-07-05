import torch.nn
from ...backpropextension import BackpropExtension
from ...gradient.linear import GradLinear
from ...extensions import VARIANCE
from ..sumgradsquared.linear import SGSLinear
from .base import variance_from


class VarianceLinear(BackpropExtension):

    def __init__(self):
        super().__init__(
            torch.nn.Linear, VARIANCE,
            req_inputs=[0], req_output=True
        )

    def apply(self, module, grad_input, grad_output):
        N = grad_output[0].shape[0]

        if module.bias is not None and module.bias.requires_grad:
            module.bias.variance = self.bias_var(module, grad_output, N)
        if module.weight.requires_grad:
            module.weight.variance = self.weight_var(module, grad_output, N)

    def bias_var(self, module, grad_output, N):
        return variance_from(
            GradLinear().bias_grad(module, grad_output),
            SGSLinear().bias_sum_grad_squared(module, grad_output),
            N
        )

    def weight_var(self, module, grad_output, N):
        return variance_from(
            GradLinear().weight_grad(module, grad_output),
            SGSLinear().weight_sum_grad_squared(module, grad_output),
            N
        )


EXTENSIONS = [VarianceLinear()]
