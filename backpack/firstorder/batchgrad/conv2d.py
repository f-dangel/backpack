import torch.nn
from ...utils.utils import einsum
from ...utils import conv as convUtils
from ..firstorder import FirstOrderExtension
from ...extensions import BATCH_GRAD


class BatchGradConv2d(FirstOrderExtension):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d, BATCH_GRAD, params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        return grad_output[0].sum(3).sum(2)

    def weight(self, module, grad_input, grad_output):
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, grad_output[0], module)

        batch = module.input0.size(0)
        dE_dw_shape = (batch, ) + module.weight.size()

        return einsum('bml,bkl->bmk', (dE_dY, X)).view(dE_dw_shape)


EXTENSIONS = [BatchGradConv2d()]
