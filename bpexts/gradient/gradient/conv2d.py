import torch.nn
from ...utils import einsum
from ..utils import unfold_func
from ..backpropextension import BackpropExtension


class GradConv2d(BackpropExtension):

    def __init__(self):
        super().__init__(
            torch.nn.Conv2d, "GRAD",
            req_inputs=[0], req_output=True
        )

    def apply(self, module, grad_input, grad_output):
        if module.bias is not None and module.bias.requires_grad:
            module.bias.grad = self.bias_grad(module, grad_output)
        if module.weight.requires_grad:
            module.weight.grad = self.weight_grad(module, grad_output)

    def bias_grad(self, module, grad_output):
        return grad_output[0].sum(3).sum(2).sum(0)

    def weight_grad(self, module, grad_output):
        batch = module.input0.size(0)
        X = unfold_func(module)(module.input0)
        dE_dY = grad_output[0].view(batch, module.out_channels, -1)
        dE_dW = einsum('bml,bkl->mk', (dE_dY, X))
        return dE_dW.view(module.weight.size())


EXTENSIONS = [GradConv2d()]
