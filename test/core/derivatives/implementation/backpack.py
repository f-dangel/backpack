"""Contains derivative calculation with BackPACK."""
from test.core.derivatives.implementation.base import DerivativesImplementation

import torch
from torch import Tensor


class BackpackDerivatives(DerivativesImplementation):
    """Derivative implementations with BackPACK."""

    def __init__(self, problem):
        """Initialization.

        Args:
            problem: test problem
        """
        problem.extend()
        super().__init__(problem)

    def store_forward_io(self):
        """Do one forward pass.

        This implicitly saves relevant quantities for backward pass.
        """
        self.problem.forward_pass()

    def jac_mat_prod(self, mat):  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.jac_mat_prod(
            self.problem.module, None, None, mat
        )

    def jac_t_mat_prod(self, mat):  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.jac_t_mat_prod(
            self.problem.module, None, None, mat
        )

    def weight_jac_t_mat_prod(self, mat, sum_batch, subsampling=None):  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.weight_jac_t_mat_prod(
            self.problem.module,
            None,
            None,
            mat,
            sum_batch=sum_batch,
            subsampling=subsampling,
        )

    def bias_jac_t_mat_prod(self, mat, sum_batch):  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.bias_jac_t_mat_prod(
            self.problem.module, None, None, mat, sum_batch=sum_batch
        )

    def weight_jac_mat_prod(self, mat):  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.weight_jac_mat_prod(
            self.problem.module, None, None, mat
        )

    def bias_jac_mat_prod(self, mat):  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.bias_jac_mat_prod(
            self.problem.module, None, None, mat
        )

    def bias_ih_l0_jac_t_mat_prod(self, mat, sum_batch):  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.bias_ih_l0_jac_t_mat_prod(
            self.problem.module, None, None, mat, sum_batch=sum_batch
        )

    def bias_hh_l0_jac_t_mat_prod(self, mat, sum_batch):  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.bias_hh_l0_jac_t_mat_prod(
            self.problem.module, None, None, mat, sum_batch=sum_batch
        )

    def weight_ih_l0_jac_t_mat_prod(self, mat, sum_batch):  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.weight_ih_l0_jac_t_mat_prod(
            self.problem.module, None, None, mat, sum_batch=sum_batch
        )

    def weight_hh_l0_jac_t_mat_prod(self, mat, sum_batch):  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.weight_hh_l0_jac_t_mat_prod(
            self.problem.module, None, None, mat, sum_batch=sum_batch
        )

    def ea_jac_t_mat_jac_prod(self, mat):  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.ea_jac_t_mat_jac_prod(
            self.problem.module, None, None, mat
        )

    def sum_hessian(self):  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.sum_hessian(self.problem.module, None, None)

    def input_hessian_via_sqrt_hessian(self, mc_samples=None) -> Tensor:
        """Computes the input hessian.

        Args:
            mc_samples: If int, uses an MC approximation with the specified
                number of samples. If None, uses the exact hessian. Defaults to None.

        Returns:
            hessian
        """
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

    def hessian_is_zero(self) -> bool:  # noqa: D102
        return self.problem.derivative.hessian_is_zero()

    def _sample_hessians_from_sqrt(self, sqrt):
        """Convert individual matrix square root into individual full matrix.

        Args:
            sqrt: individual square root of hessian

        Returns:
            individual full matrix

        Raises:
            ValueError: if input is not 3d
        """
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
