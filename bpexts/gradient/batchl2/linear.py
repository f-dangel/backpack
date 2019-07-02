import torch.nn
from ...utils import einsum
from ..backpropextension import BackpropExtension
from ..extensions import BATCH_L2


class BatchL2Linear(BackpropExtension):

    def __init__(self):
        super().__init__(
            torch.nn.Linear, BATCH_L2,
            req_inputs=[0], req_output=True
        )

    def apply(self, module, grad_input, grad_output):
        if module.bias is not None and module.bias.requires_grad:
            module.bias.batch_l2 = self.bias(module, grad_output)
        if module.weight.requires_grad:
            module.weight.batch_l2 = self.weight(module, grad_output)

    def bias(self, module, grad_output):
        return (grad_output[0]**2).sum(list(range(1, len(grad_output[0].shape))))

    def weight(self, module, grad_output):
        return einsum('bi,bj->b', (grad_output[0]**2, module.input0**2))


EXTENSIONS = [BatchL2Linear()]
