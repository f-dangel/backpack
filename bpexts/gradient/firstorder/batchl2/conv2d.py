import torch.nn
from ....utils import einsum
from ...utils import conv as convUtils
from ...backpropextension import BackpropExtension
from ...extensions import BATCH_L2


class BatchL2Conv2d(BackpropExtension):

    def __init__(self):
        super().__init__(
            torch.nn.Conv2d, BATCH_L2,
            req_inputs=[0], req_output=True
        )

    def apply(self, module, grad_input, grad_output):
        if module.bias is not None and module.bias.requires_grad:
            module.bias.batch_l2 = self.bias(module, grad_output)
        if module.weight.requires_grad:
            module.weight.batch_l2 = self.weight(module, grad_output)

    def bias(self, module, grad_output):
        elems_squared = (grad_output[0].sum(3).sum(2)**2)
        return elems_squared.sum(list(range(1, len(elems_squared.shape))))

    def weight(self, module, grad_output):
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, grad_output[0], module
        )
        w_l2 = einsum('bml,bkl,bmi,bki->b', (dE_dY, X, dE_dY, X))
        return w_l2


EXTENSIONS = [BatchL2Conv2d()]
