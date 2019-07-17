import torch.nn
from ...utils.utils import einsum
from ...extensions import SUM_GRAD_SQUARED
from ..firstorder import FirstOrderExtension
from ...core.layers import LinearConcat


class SGSLinear(FirstOrderExtension):
    def __init__(self):
        super().__init__(
            torch.nn.Linear, SUM_GRAD_SQUARED, params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        return (grad_output[0]**2).sum(0)

    def weight(self, module, grad_input, grad_output):
        return einsum('bi,bj->ij', (grad_output[0]**2, module.input0**2))


class SGSLinearConcat(FirstOrderExtension):
    def __init__(self):
        super().__init__(LinearConcat, SUM_GRAD_SQUARED, params=["weight"])

    def weight(self, module, grad_input, grad_output):
        input = module.input0
        if module.has_bias():
            input = module.append_ones(input)
        return einsum('bi,bj->ij', (grad_output[0]**2, input**2))


EXTENSIONS = [SGSLinear(), SGSLinearConcat()]
