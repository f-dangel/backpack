import torch.nn
from ....utils import einsum
from ...extensions import GRAD
from ..firstorder import FirstOrderExtension


class GradLinear(FirstOrderExtension):

    def __init__(self):
        super().__init__(torch.nn.Linear, GRAD, params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        return grad_output[0].sum(0)

    def weight(self, module, grad_input, grad_output):
        w_grad_batch = einsum('bi,bj->ij', (grad_output[0], module.input0))
        return w_grad_batch.view(module.out_features, module.in_features)
