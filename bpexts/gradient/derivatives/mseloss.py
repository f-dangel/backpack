from math import sqrt
from torch import diag_embed, ones_like
from torch.nn import MSELoss
from .basederivatives import BaseDerivatives


class MSELossDerivatives(BaseDerivatives):
    def get_module(self):
        return MSELoss

    def sqrt_hessian(self, module, grad_input, grad_output):
        self.check_input_dims(module)

        sqrt_H = diag_embed(sqrt(2) * ones_like(module.input0))

        if module.reduction is "mean":
            sqrt_H /= sqrt(module.input0.shape[0])

        return sqrt_H

    def sum_hessian(self, module, grad_input, grad_output):
        self.check_input_dims(module)

        sum_H = diag_embed(2 * ones_like(module.input0))

        if module.reduction is "mean":
            sum_H /= module.input0.shape[0]

        return sum_H

    def check_input_dims(module):
        if not len(module.input0.shape) == 2:
            raise ValueError(
                "Only 2D inputs are currently supported for MSELoss.")
