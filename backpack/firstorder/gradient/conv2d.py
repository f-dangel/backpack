import torch.nn
from ...core.layers import Conv2dConcat
from ...core.derivatives.conv2d import (Conv2DDerivatives,
                                        Conv2DConcatDerivatives)
from ...extensions import GRAD
from ..firstorder import FirstOrderExtension


class GradConv2d(FirstOrderExtension, Conv2DDerivatives):
    def __init__(self):
        super().__init__(torch.nn.Conv2d, GRAD, params=["bias", "weight"])

    # TODO: Same code as for batch gradient, but with sum_batch = True
    def bias(self, module, grad_input, grad_output):
        batch = grad_output[0].shape[0]
        shape = module.bias.shape

        grad_out_vec = grad_output[0].contiguous().view(batch, -1)

        bias_grad = self.bias_jac_t_mat_prod(
            module, grad_input, grad_output, grad_out_vec, sum_batch=True)

        return bias_grad.view(shape)

    # TODO: Same code as for batch gradient, but with sum_batch = True
    def weight(self, module, grad_input, grad_output):
        batch = grad_output[0].shape[0]
        shape = module.weight.shape

        grad_out_vec = grad_output[0].contiguous().view(batch, -1)

        weight_grad = self.weight_jac_t_mat_prod(
            module, grad_input, grad_output, grad_out_vec, sum_batch=True)

        return weight_grad.view(shape)


class GradConv2dConcat(FirstOrderExtension, Conv2DConcatDerivatives):
    def __init__(self):
        super().__init__(Conv2dConcat, GRAD, params=["weight"])

    # TODO: Same code as for batch gradient, but with sum_batch = True
    def weight(self, module, grad_input, grad_output):
        batch = grad_output[0].shape[0]
        shape = module.weight.shape

        grad_out_vec = grad_output[0].contiguous().view(batch, -1)

        weight_grad = self.weight_jac_t_mat_prod(
            module, grad_input, grad_output, grad_out_vec, sum_batch=True)

        return weight_grad.view(shape)
