import torch.nn
from ....utils import einsum
from ..firstorder import FirstOrderExtension
from ...extensions import SUM_GRAD_SQUARED


class SGSLinear(FirstOrderExtension):

    def __init__(self):
        super().__init__(torch.nn.Linear, SUM_GRAD_SQUARED, params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        return (grad_output[0]**2).sum(0)

    def weight(self, module, grad_input, grad_output):
        return einsum('bi,bj->ij', (grad_output[0]**2, module.input0**2))


EXTENSIONS = [SGSLinear()]
