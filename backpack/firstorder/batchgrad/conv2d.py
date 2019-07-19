import torch.nn
from ...utils.utils import einsum
from ...utils import conv as convUtils
from ...core.layers import Conv2dConcat
from ..firstorder import FirstOrderExtension
from ...extensions import BATCH_GRAD
from ...core.derivatives.conv2d import (Conv2DDerivatives,
                                        Conv2DConcatDerivatives)


class BatchGradConv2d(FirstOrderExtension, Conv2DDerivatives):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d, BATCH_GRAD, params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        batch = grad_output[0].shape[0]
        shape = (batch, *module.bias.shape)

        grad_out_vec = grad_output[0].contiguous().view(batch, -1)

        bias_grad = self.bias_jac_t_mat_prod(
            module, grad_input, grad_output, grad_out_vec, sum_batch=False)

        return bias_grad.view(shape)

    def weight(self, module, grad_input, grad_output):
        batch = grad_output[0].shape[0]
        shape = (batch, *module.weight.shape)

        grad_out_vec = grad_output[0].contiguous().view(batch, -1)

        weight_grad = self.weight_jac_t_mat_prod(
            module, grad_input, grad_output, grad_out_vec, sum_batch=False)

        return weight_grad.view(shape)


class BatchGradConv2dConcat(FirstOrderExtension):
    def __init__(self):
        super().__init__(Conv2dConcat, BATCH_GRAD, params=["weight"])

    def weight(self, module, grad_input, grad_output):
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, grad_output[0], module)

        if module.has_bias():
            X = module.append_ones(X)

        batch = module.input0.size(0)
        dE_dw_shape = (batch, ) + module.weight.size()

        return einsum('bml,bkl->bmk', (dE_dY, X)).view(dE_dw_shape)


EXTENSIONS = [BatchGradConv2d(), BatchGradConv2dConcat()]
