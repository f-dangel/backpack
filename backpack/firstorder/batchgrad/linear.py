import torch.nn
from ...utils.utils import einsum
from ..firstorder import FirstOrderExtension
from ...extensions import BATCH_GRAD
from ...core.layers import LinearConcat

from ...core.derivatives.linear import (LinearDerivatives,
                                        LinearConcatDerivatives)


class BatchGradLinear(FirstOrderExtension, LinearDerivatives):
    def __init__(self):
        super().__init__(
            torch.nn.Linear, BATCH_GRAD, params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        batch = grad_output[0].shape[0]
        shape = (batch, *module.bias.shape)

        weight_grad = self.bias_jac_t_mat_prod(
            module, grad_input, grad_output, grad_output[0], sum_batch=False)

        return weight_grad.view(shape)

    def weight(self, module, grad_input, grad_output):
        batch = grad_output[0].shape[0]
        shape = (batch, *module.weight.shape)

        weight_grad = self.weight_jac_t_mat_prod(
            module, grad_input, grad_output, grad_output[0], sum_batch=False)

        return weight_grad.view(shape)


class BatchGradLinearConcat(FirstOrderExtension, LinearConcatDerivatives):
    def __init__(self):
        super().__init__(LinearConcat, BATCH_GRAD, params=["weight"])

    def weight(self, module, grad_input, grad_output):
        batch = grad_output[0].shape[0]
        shape = (batch, *module.weight.shape)

        weight_grad = self.weight_jac_t_mat_prod(
            module, grad_input, grad_output, grad_output[0], sum_batch=False)

        return weight_grad.view(shape)


EXTENSIONS = [BatchGradLinear(), BatchGradLinearConcat()]
