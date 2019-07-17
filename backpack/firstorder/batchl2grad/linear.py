import torch.nn
from ...utils.utils import einsum
from ..firstorder import FirstOrderExtension
from ...extensions import BATCH_L2
from ...core.layers import LinearConcat


class BatchL2Linear(FirstOrderExtension):
    def __init__(self):
        super().__init__(torch.nn.Linear, BATCH_L2, params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        return (grad_output[0]**2).sum(1)

    def weight(self, module, grad_input, grad_output):
        return einsum('bi,bj->b', (grad_output[0]**2, module.input0**2))


class BatchL2ConcatLinear(FirstOrderExtension):
    def __init__(self):
        super().__init__(LinearConcat, BATCH_L2, params=["weight"])

    def weight(self, module, grad_input, grad_output):
        input = module.input0
        if module.has_bias():
            input = module.append_ones(input)
        return einsum('bi,bj->b', (grad_output[0]**2, input**2))


EXTENSIONS = [BatchL2Linear(), BatchL2ConcatLinear()]
