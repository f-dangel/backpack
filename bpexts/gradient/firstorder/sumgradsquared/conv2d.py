import torch.nn
from ....utils import einsum
from ...utils import conv as convUtils
from ...backpropextension import BackpropExtension
from ...extensions import SUM_GRAD_SQUARED


class SGSConv2d(BackpropExtension):

    def __init__(self):
        super().__init__(
            torch.nn.Conv2d, SUM_GRAD_SQUARED,
            req_inputs=[0], req_output=True
        )

    def apply(self, module, grad_input, grad_output):
        """Compute sum of squared batch gradients of `torch.nn.Conv2d` parameters.

        - Store bias sum of squared gradients in `module.bias.sum_grad_squared`
        - Store kernel sum of squared gradients in `module.weight.sum_grad_squared`
        """
        if module.bias is not None and module.bias.requires_grad:
            module.bias.sum_grad_squared = self.bias_sum_grad_squared(
                module, grad_output)
        if module.weight.requires_grad:
            module.weight.sum_grad_squared = self.weight_sum_grad_squared(
                module, grad_output)

    def bias_sum_grad_squared(self, module, grad_output):
        return (grad_output[0].sum(3).sum(2)**2).sum(0)

    def weight_sum_grad_squared(self, module, grad_output):
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, grad_output[0], module
        )
        d1 = einsum('bml,bkl->bmk', (dE_dY, X))
        d2 = (d1**2).sum(0)
        return d2.view_as(module.weight)


EXTENSIONS = [SGSConv2d()]
