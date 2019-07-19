import torch.nn
from ...core.layers import LinearConcat
from ...core.derivatives.linear import (LinearDerivatives,
                                        LinearConcatDerivatives)
from ...utils.utils import einsum
from ...extensions import GRAD
from ..firstorder import FirstOrderExtension


class GradLinear(FirstOrderExtension, LinearDerivatives):
    def __init__(self):
        super().__init__(torch.nn.Linear, GRAD, params=["bias", "weight"])

    # TODO: Same code as for batch gradient, but with sum_batch = True
    def bias(self, module, grad_input, grad_output):
        shape = module.bias.shape

        bias_grad = self.bias_jac_t_mat_prod(
            module, grad_input, grad_output, grad_output[0], sum_batch=True)

        return bias_grad.view(shape)

    # TODO: Same code as for batch gradient, but with sum_batch = True
    def weight(self, module, grad_input, grad_output):
        shape = module.weight.shape

        weight_grad = self.weight_jac_t_mat_prod(
            module, grad_input, grad_output, grad_output[0], sum_batch=True)

        return weight_grad.view(shape)


class GradLinearConcat(FirstOrderExtension, LinearConcatDerivatives):
    def __init__(self):
        super().__init__(LinearConcat, GRAD, params=["weight"])

    # TODO: Same code as for batch gradient, but with sum_batch = True
    def weight(self, module, grad_input, grad_output):
        shape = module.weight.shape

        weight_grad = self.weight_jac_t_mat_prod(
            module, grad_input, grad_output, grad_output[0], sum_batch=True)

        return weight_grad.view(shape)
