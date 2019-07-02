import torch.nn
from ..backpropextension import BackpropExtension
from ..sumgradsquared.conv2d import SGSConv2d
from ..gradient.conv2d import GradConv2d


class VarianceConv2d(BackpropExtension):

    def __init__(self):
        super().__init__(
            torch.nn.Conv2d, "VARIANCE",
            req_inputs=[0], req_output=True
        )

    def apply(self, module, grad_input, grad_output):
        if module.bias is not None and module.bias.requires_grad:
            module.bias.variance = self.bias(module, grad_output)
        if module.weight.requires_grad:
            module.weight.variance = self.weight(module, grad_output)

    def bias(self, module, grad_output):
        N = grad_output[0].shape[0]
        avgg_squared = (GradConv2d().bias_grad(module, grad_output) / N)**2
        avg_gsquared = SGSConv2d().bias_sum_grad_squared(module, grad_output) / N
        return avg_gsquared - avgg_squared

    def weight(self, module, grad_output):
        N = grad_output[0].shape[0]
        avgg_squared = (GradConv2d().weight_grad(module, grad_output) / N)**2
        avg_gsquared = SGSConv2d().weight_sum_grad_squared(module, grad_output) / N
        return avg_gsquared - avgg_squared


EXTENSIONS = [VarianceConv2d()]
