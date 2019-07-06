from math import sqrt
from torch import diag_embed, ones_like
from torch.nn import MSELoss
from .basederivatives import BaseDerivatives


class MSELossDerivatives(BaseDerivatives):

    def get_module(self):
        return MSELoss

    def sqrt_hessian(self, module, grad_input, grad_output):
        if not len(module.input0.shape) == 2:
            raise ValueError(
                "Only 2D inputs are currently supported for MSELoss.")

        sqrt_h = diag_embed(sqrt(2) * ones_like(module.input0))

        if module.reduction is "mean":
            sqrt_h /= sqrt(module.input0.shape[0])

        return sqrt_h
