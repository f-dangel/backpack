import torch.nn
from ...core.layers import Conv2dConcat
from ...utils.utils import einsum
from ...utils import conv as convUtils
from ..firstorder import FirstOrderExtension
from ...extensions import BATCH_L2


class BatchL2Conv2d(FirstOrderExtension):
    def __init__(self):
        super().__init__(torch.nn.Conv2d, BATCH_L2, params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        return (grad_output[0].sum(3).sum(2)**2).sum(1)

    def weight(self, module, grad_input, grad_output):
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, grad_output[0], module)
        return einsum('bml,bkl,bmi,bki->b', (dE_dY, X, dE_dY, X))


class BatchL2Conv2dConcat(FirstOrderExtension):
    def __init__(self):
        super().__init__(Conv2dConcat, BATCH_L2, params=["weight"])

    def weight(self, module, grad_input, grad_output):
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, grad_output[0], module)

        if module.has_bias():
            X = module.append_ones(X)

        return einsum('bml,bkl,bmi,bki->b', (dE_dY, X, dE_dY, X))


EXTENSIONS = [BatchL2Conv2d(), BatchL2Conv2dConcat()]
