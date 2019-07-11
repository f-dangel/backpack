import torch.nn
from ....utils import einsum
from ...utils import conv as convUtils
from ...extensions import SUM_GRAD_SQUARED
from ..firstorder import FirstOrderExtension


class SGSConv2d(FirstOrderExtension):

    def __init__(self):
        super().__init__(torch.nn.Conv2d, SUM_GRAD_SQUARED, params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        return (grad_output[0].sum(3).sum(2)**2).sum(0)

    def weight(self, module, grad_input, grad_output):
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, grad_output[0], module
        )
        d1 = einsum('bml,bkl->bmk', (dE_dY, X))
        return (d1**2).sum(0).view_as(module.weight)


EXTENSIONS = [SGSConv2d()]
