import torch.nn
from ...utils import einsum
from ..utils import unfold_func
from ..batchgrad.conv2d import BatchGradConv2d


class SGSConv2d(BatchGradConv2d):

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
        """Compute sum of squared batch gradients for bias."""
        return (self.bias_grad_batch(module, grad_output)**2).sum(0)

    def weight_sum_grad_squared(self, module, grad_output):
        """Compute sum of squared batch gradients for kernel."""
        batch = module.input0.size(0)
        X = unfold_func(module)(module.input0)
        dE_dY = grad_output[0].view(batch, module.out_channels, -1)
        w_sgs = einsum('bml,bkl,bmi,bki->mk', (dE_dY, X, dE_dY, X))
        return w_sgs.view_as(module.weight)


SIGNATURE = [(torch.nn.Conv2d, "SUM_GRAD_SQUARED", SGSConv2d())]
