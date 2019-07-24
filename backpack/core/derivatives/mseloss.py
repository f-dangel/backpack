from warnings import warn
from math import sqrt
from torch import diag_embed, ones_like, randn, diag, ones
from torch.nn import MSELoss
from .basederivatives import BaseDerivatives

from .utils import hmp_unsqueeze_if_missing_dim


class MSELossDerivatives(BaseDerivatives):
    def get_module(self):
        return MSELoss

    def sqrt_hessian(self, module, g_inp, g_out):
        self.check_input_dims(module)

        sqrt_H = diag_embed(sqrt(2) * ones_like(module.input0))

        if module.reduction is "mean":
            sqrt_H /= sqrt(module.input0.shape[0])

        return sqrt_H

    def sqrt_hessian_sampled(self, module, g_inp, g_out):
        warn("[MC Sampling Hessian of CrossEntropy] " +
             "Returning the symmetric factorization of the full Hessian " +
             "(same computation cost)")
        return self.sqrt_hessian(module, g_inp, g_out)

    def sum_hessian(self, module, g_inp, g_out):
        self.check_input_dims(module)

        batch = module.input0_shape[0]
        num_features = module.input0.numel() // batch
        sum_H = 2 * batch * diag(
            ones(num_features, device=module.input0.device))

        if module.reduction is "mean":
            sum_H /= module.input0.shape[0]
        print("sum H ", sum_H.shape)
        return sum_H

    def hessian_matrix_product(self, module, g_inp, g_out):
        """Multiplication of the input Hessian with a matrix."""

        @hmp_unsqueeze_if_missing_dim(mat_dim=3)
        def hmp(mat):
            Hmat = 2 * mat

            if module.reduction is "mean":
                Hmat /= module.input0.shape[0]

            return Hmat

        return hmp

    def check_input_dims(self, module):
        if not len(module.input0.shape) == 2:
            raise ValueError(
                "Only 2D inputs are currently supported for MSELoss.")

    def hessian_is_psd(self):
        return True
