from math import sqrt
from warnings import warn

from torch import diag, diag_embed, ones, ones_like
from torch.nn import MSELoss

from backpack.core.derivatives.basederivatives import BaseLossDerivatives


class MSELossDerivatives(BaseLossDerivatives):
    def get_module(self):
        return MSELoss

    def _sqrt_hessian(self, module, g_inp, g_out):
        self.check_input_dims(module)

        V_dim, C_dim = 0, 2
        diag = sqrt(2) * ones_like(module.input0)
        sqrt_H = diag_embed(diag, dim1=V_dim, dim2=C_dim)

        if module.reduction == "mean":
            sqrt_H /= sqrt(self._mean_factor(module))

        return sqrt_H

    def _sqrt_hessian_sampled(self, module, g_inp, g_out, mc_samples=None):
        """
        Note:
        -----
        The parameter `mc_samples` is ignored.
        The method always returns the full square root.

        The computational cost between the sampled and full version is the same,
        so the method always return the more accurate version.

        The cost is the same because the hessian of the loss w.r.t. its inputs
        for a single sample is one-dimensional.
        """
        N = module.input0_shape[0]
        input_numel = self._mean_factor(module)
        num_features = input_numel // N

        if num_features == 1:
            warn(
                "[MC Sampling Hessian of MSE loss] "
                + "Returning the symmetric factorization of the full Hessian "
                + "(same computation cost)",
                UserWarning,
            )
            return self.sqrt_hessian(module, g_inp, g_out)
        else:
            raise NotImplementedError(
                "No sampling supported for features >1 (got {})".format(input_numel)
            )

    def _sum_hessian(self, module, g_inp, g_out):
        self.check_input_dims(module)

        N = module.input0_shape[0]
        num_features = module.input0.numel() // N
        sum_H = 2 * N * diag(ones(num_features, device=module.input0.device))

        if module.reduction == "mean":
            sum_H /= self._mean_factor(module)

        return sum_H

    def _make_hessian_mat_prod(self, module, g_inp, g_out):
        """Multiplication of the input Hessian with a matrix."""

        def hessian_mat_prod(mat):
            Hmat = 2 * mat

            if module.reduction == "mean":
                Hmat /= self._mean_factor(module)

            return Hmat

        return hessian_mat_prod

    def _mean_factor(self, module):
        """Factor for division with reduction 'mean'."""
        return module.input0.numel()

    def check_input_dims(self, module):
        if not len(module.input0.shape) == 2:
            raise ValueError("Only 2D inputs are currently supported for MSELoss.")

    def hessian_is_psd(self):
        return True
