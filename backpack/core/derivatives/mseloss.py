"""Derivatives of the MSE Loss."""

from math import sqrt

from torch import einsum, eye, normal

from backpack.core.derivatives.basederivatives import BaseLossDerivatives


class MSELossDerivatives(BaseLossDerivatives):
    """Derivatives of the MSE Loss.

    We only support 2D tensors.

    For `X : [n, d]` and `Y : [n, d]`, if `reduce=sum`, the MSE computes
    `∑ᵢ₌₁ⁿ ‖X[i,∶] − Y[i,∶]‖²`. If `reduce=mean`, the result is divided by `nd`.
    """

    def _sqrt_hessian(self, module, g_inp, g_out):
        """Square-root of the hessian of the MSE for each minibatch elements.

        Returns the Hessian in format `Hs = [D, N, D]`, where
        `Hs[:, n, :]` is the Hessian for the `n`th element.

        Attributes:
            module: (torch.nn.MSELoss) module
            g_inp: Gradient of loss w.r.t. input
            g_out: Gradient of loss w.r.t. output

        Returns:
             Batch of hessians, in format [D, N, D]
        """
        self.check_input_dims(module)

        N, D = module.input0.shape
        sqrt_H = sqrt(2) * eye(D, device=module.input0.device)  # [D, D]
        sqrt_H = sqrt_H.unsqueeze(0).repeat(N, 1, 1)  # [N, D, D]
        sqrt_H = einsum("nab->anb", sqrt_H)  # [D, N, D]

        if module.reduction == "mean":
            sqrt_H /= sqrt(module.input0.numel())

        return sqrt_H

    def _sqrt_hessian_sampled(self, module, g_inp, g_out, mc_samples=1):
        """A Monte-Carlo estimate of the square-root of the Hessian.

        Attributes:
            module: (torch.nn.MSELoss) module.
            g_inp: Gradient of loss w.r.t. input.
            g_out: Gradient of loss w.r.t. output.
            mc_samples: (int, optional) Number of MC samples to use. Default: 1.

        Returns:
            tensor:
        """
        N, D = module.input0.shape
        samples = normal(0, 1, size=[mc_samples, N, D], device=module.input0.device)
        samples *= sqrt(2) / sqrt(mc_samples)

        if module.reduction == "mean":
            samples /= sqrt(module.input0.numel())

        return samples

    def _sum_hessian(self, module, g_inp, g_out):
        """The Hessian, summed across the batch dimension.

        Args:
            module: (torch.nn.MSELoss) module
            g_inp: Gradient of loss w.r.t. input
            g_out: Gradient of loss w.r.t. output

        Returns: a `[D, D]` tensor of the Hessian, summed across batch

        """
        self.check_input_dims(module)

        N, D = module.input0.shape
        H = 2 * eye(D, device=module.input0.device)

        if module.reduction == "sum":
            H *= N
        elif module.reduction == "mean":
            H /= D

        return H

    def _make_hessian_mat_prod(self, module, g_inp, g_out):
        """Multiplication of the input Hessian with a matrix."""

        def hessian_mat_prod(mat):
            Hmat = 2 * mat
            if module.reduction == "mean":
                Hmat /= module.input0.numel()
            return Hmat

        return hessian_mat_prod

    def check_input_dims(self, module):
        """Raises an exception if the shapes of the input are not supported."""
        if not len(module.input0.shape) == 2:
            raise ValueError("Only 2D inputs are currently supported for MSELoss.")

    def hessian_is_psd(self):
        return True
