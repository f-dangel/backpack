from test.core.derivatives.implementation.base import DerivativesImplementation

import torch


class BackpackDerivatives(DerivativesImplementation):
    """Derivative implementations with BackPACK."""

    def __init__(self, problem):
        problem.extend()
        super().__init__(problem)

    def store_forward_io(self):
        self.problem.forward_pass()

    def jac_mat_prod(self, mat):
        self.store_forward_io()
        return self.problem.derivative.jac_mat_prod(
            self.problem.module, None, None, mat
        )

    def jac_t_mat_prod(self, mat):
        self.store_forward_io()
        return self.problem.derivative.jac_t_mat_prod(
            self.problem.module, None, None, mat
        )

    def weight_jac_t_mat_prod(self, mat, sum_batch):
        self.store_forward_io()
        return self.problem.derivative.weight_jac_t_mat_prod(
            self.problem.module, None, None, mat, sum_batch=sum_batch
        )

    def bias_jac_t_mat_prod(self, mat, sum_batch):
        self.store_forward_io()
        return self.problem.derivative.bias_jac_t_mat_prod(
            self.problem.module, None, None, mat, sum_batch=sum_batch
        )

    def weight_jac_mat_prod(self, mat):
        self.store_forward_io()
        return self.problem.derivative.weight_jac_mat_prod(
            self.problem.module, None, None, mat
        )

    def bias_jac_mat_prod(self, mat):
        self.store_forward_io()
        return self.problem.derivative.bias_jac_mat_prod(
            self.problem.module, None, None, mat
        )

    def ea_jac_t_mat_jac_prod(self, mat):
        self.store_forward_io()
        return self.problem.derivative.ea_jac_t_mat_jac_prod(
            self.problem.module, None, None, mat
        )

    def sum_hessian(self):
        self.store_forward_io()
        return self.problem.derivative.sum_hessian(self.problem.module, None, None)

    def input_hessian_via_sqrt_hessian(self, mc_samples=None):
        # MC_SAMPLES = 100000
        self.store_forward_io()

        if mc_samples is not None:
            sqrt_hessian = self.problem.derivative.sqrt_hessian_sampled(
                self.problem.module, None, None, mc_samples=mc_samples
            )
        else:
            sqrt_hessian = self.problem.derivative.sqrt_hessian(
                self.problem.module, None, None
            )

        individual_hessians = self._sample_hessians_from_sqrt(sqrt_hessian)

        return self._embed_sample_hessians(
            individual_hessians, self.problem.module.input0
        )

    def hessian_is_zero(self):
        """Return whether the input-output Hessian is zero.

        Returns:
            bool: `True`, if Hessian is zero, else `False`.
        """
        return self.problem.derivative.hessian_is_zero()

    def _sample_hessians_from_sqrt(self, sqrt):
        """Convert individual matrix square root into individual full matrix. """
        equation = None
        num_axes = len(sqrt.shape)

        # TODO improve readability
        if num_axes == 3:
            equation = "vni,vnj->nij"
        else:
            raise ValueError("Only 2D inputs are currently supported.")

        return torch.einsum(equation, sqrt, sqrt)

    def _embed_sample_hessians(self, individual_hessians, input):
        hessian_shape = (*input.shape, *input.shape)
        hessian = torch.zeros(hessian_shape, device=input.device)

        N = input.shape[0]

        for n in range(N):
            num_axes = len(input.shape)

            if num_axes == 2:
                hessian[n, :, n, :] = individual_hessians[n]
            else:
                raise ValueError("Only 2D inputs are currently supported.")

        return hessian
