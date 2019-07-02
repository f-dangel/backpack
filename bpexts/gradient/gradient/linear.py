import torch.nn
from ...utils import einsum
from ..backpropextension import BackpropExtension
from ..extensions import GRAD


class GradLinear(BackpropExtension):

    def __init__(self):
        super().__init__(
            torch.nn.Linear, GRAD,
            req_inputs=[0], req_output=True
        )

    def apply(self, module, grad_input, grad_output):
        if module.bias is not None and module.bias.requires_grad:
            module.bias.grad_batch = self.bias_grad(module, grad_output)
        if module.weight.requires_grad:
            module.weight.grad_batch = self.weight_grad(module, grad_output)

    def bias_grad(self, module, grad_output):
        return grad_output[0].sum(0)

    def weight_grad(self, module, grad_output):
        w_grad_batch = einsum('bi,bj->ij', (grad_output[0], module.input0))
        return w_grad_batch.view(module.out_features, module.in_features)
