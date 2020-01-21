from warnings import warn
from math import sqrt
from torch import diag_embed, ones_like, diag, ones
from torch.nn import MSELoss
from backpack.core.derivatives.basederivatives import BaseDerivatives

from backpack.core.derivatives.utils import hessian_matrix_product_accept_vectors


class MSELossDerivatives(BaseDerivatives):
    def get_module(self):
        return MSELoss

    def sqrt_hessian(self, module, g_inp, g_out):
        self.check_input_dims(module)

        V_dim, C_dim = 0, 2
        diag = sqrt(2) * ones_like(module.input0)
        sqrt_H = diag_embed(diag, dim1=V_dim, dim2=C_dim)

        if module.reduction == "mean":
            N = module.input0.shape[0]
            sqrt_H /= sqrt(N)

        return sqrt_H

    def sqrt_hessian_sampled(self, module, g_inp, g_out):
        warn(
            "[MC Sampling Hessian of MSE loss] "
            + "Returning the symmetric factorization of the full Hessian "
            + "(same computation cost)",
            UserWarning,
        )
        return self.sqrt_hessian(module, g_inp, g_out)

    def sum_hessian(self, module, g_inp, g_out):
        self.check_input_dims(module)

        N = module.input0_shape[0]
        num_features = module.input0.numel() // N
        sum_H = 2 * N * diag(ones(num_features, device=module.input0.device))

        if module.reduction == "mean":
            sum_H /= N

        return sum_H

    @hessian_matrix_product_accept_vectors
    def hessian_matrix_product(self, module, g_inp, g_out):
        """Multiplication of the input Hessian with a matrix."""

        def hmp(mat):
            Hmat = 2 * mat

            if module.reduction == "mean":
                N = module.input0.shape[0]
                Hmat /= N

            return Hmat

        return hmp

    def check_input_dims(self, module):
        if not len(module.input0.shape) == 2:
            raise ValueError("Only 2D inputs are currently supported for MSELoss.")

    def hessian_is_psd(self):
        return True
