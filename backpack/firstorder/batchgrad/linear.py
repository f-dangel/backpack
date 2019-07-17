import torch.nn
from ...utils.utils import einsum
from ..firstorder import FirstOrderExtension
from ...extensions import BATCH_GRAD
from ...core.layers import LinearConcat


class BatchGradLinear(FirstOrderExtension):
    def __init__(self):
        super().__init__(
            torch.nn.Linear, BATCH_GRAD, params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        return grad_output[0]

    def weight(self, module, grad_input, grad_output):
        return einsum('bi,bj->bij', (grad_output[0], module.input0))


class BatchGradLinearConcat(FirstOrderExtension):
    def __init__(self):
        super().__init__(LinearConcat, BATCH_GRAD, params=["weight"])

    def weight(self, module, grad_input, grad_output):
        input = module.homogeneous_input()
        return einsum('bi,bj->bij', (grad_output[0], input))


EXTENSIONS = [BatchGradLinear(), BatchGradLinearConcat()]
