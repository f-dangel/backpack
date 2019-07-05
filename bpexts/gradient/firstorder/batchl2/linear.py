import torch.nn
from ....utils import einsum
from ..firstorder import FirstOrderExtension
from ...extensions import BATCH_L2


class BatchL2Linear(FirstOrderExtension):

    def __init__(self):
        super().__init__(torch.nn.Linear, BATCH_L2, params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        return (grad_output[0]**2).sum(1)

    def weight(self, module, grad_input, grad_output):
        return einsum('bi,bj->b', (grad_output[0]**2, module.input0**2))


EXTENSIONS = [BatchL2Linear()]
