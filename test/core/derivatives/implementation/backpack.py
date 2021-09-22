"""Contains derivative calculation with BackPACK."""
from test.core.derivatives.implementation.base import DerivativesImplementation
from test.utils import chunk_sizes
from typing import List

from torch import Tensor, einsum, zeros

from backpack.utils.subsampling import subsample


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

    def jac_t_mat_prod(
        self, mat: Tensor, subsampling: List[int]
    ) -> Tensor:  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.jac_t_mat_prod(
            self.problem.module, None, None, mat, subsampling=subsampling
        )

    def param_mjp(
        self,
        param_str: str,
        mat: Tensor,
        sum_batch: bool,
        subsampling: List[int] = None,
    ) -> Tensor:  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.param_mjp(
            param_str,
            self.problem.module,
            None,
            None,
            mat,
            sum_batch=sum_batch,
            subsampling=subsampling,
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

    def ea_jac_t_mat_jac_prod(self, mat):  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.ea_jac_t_mat_jac_prod(
            self.problem.module, None, None, mat
        )

    def sum_hessian(self):  # noqa: D102
        self.store_forward_io()
        return self.problem.derivative.sum_hessian(self.problem.module, None, None)

    def input_hessian_via_sqrt_hessian(
        self, mc_samples: int = None, chunks: int = 1, subsampling: List[int] = None
    ) -> Tensor:
        """Computes the Hessian w.r.t. to the input from its matrix square root.

        Args:
            mc_samples: If int, uses an MC approximation with the specified
                number of samples. If None, uses the exact hessian. Defaults to None.
            chunks: Maximum sequential split of the computation. Default: ``1``.
                Only used if mc_samples is specified.
            subsampling: Indices of active samples. ``None`` uses all samples.

        Returns:
            Hessian with respect to the input. Has shape
            ``[N, A, B, ..., N, A, B, ...]`` where ``N`` is the batch size or number
            of active samples when sub-sampling is used, and ``[A, B, ...]`` are the
            input's feature dimensions.
        """
        self.store_forward_io()

        if mc_samples is not None:
            chunk_samples = chunk_sizes(mc_samples, chunks)
            chunk_weights = [samples / mc_samples for samples in chunk_samples]

            individual_hessians: Tensor = sum(
                weight
                * self._sample_hessians_from_sqrt(
                    self.problem.derivative.sqrt_hessian_sampled(
                        self.problem.module,
                        None,
                        None,
                        mc_samples=samples,
                        subsampling=subsampling,
                    )
                )
                for weight, samples in zip(chunk_weights, chunk_samples)
            )
        else:
            sqrt_hessian = self.problem.derivative.sqrt_hessian(
                self.problem.module, None, None, subsampling=subsampling
            )
            individual_hessians = self._sample_hessians_from_sqrt(sqrt_hessian)

        input0 = subsample(self.problem.module.input0, subsampling=subsampling)
        return self._embed_sample_hessians(individual_hessians, input0)

    def hessian_is_zero(self) -> bool:  # noqa: D102
        return self.problem.derivative.hessian_is_zero(self.problem.module)

    def _sample_hessians_from_sqrt(self, sqrt: Tensor) -> Tensor:
        """Convert individual matrix square root into individual full matrix.

        Args:
            sqrt: individual square root of hessian

        Returns:
            Individual Hessians of shape ``[N, A, B, ..., A, B, ...]`` where
            ``input.shape[1:] = [A, B, ...]`` are the input feature dimensions
            and ``N`` is the batch size.
        """
        N, input_dims = sqrt.shape[1], sqrt.shape[2:]

        sqrt_flat = sqrt.flatten(start_dim=2)
        sample_hessians = einsum("vni,vnj->nij", sqrt_flat, sqrt_flat)

        return sample_hessians.reshape(N, *input_dims, *input_dims)

    def _embed_sample_hessians(
        self, individual_hessians: Tensor, input: Tensor
    ) -> Tensor:
        """Embed Hessians w.r.t. individual samples into Hessian w.r.t. all samples.

        Args:
            individual_hessians: Hessians w.r.t. individual samples in the input.
            input: Inputs for the for samples whose individual Hessians are passed.
                Has shape ``[N, A, B, ..., A, B, ...]`` where ``N`` is the number of
                active samples and ``[A, B, ...]`` are the feature dimensions.

        Returns:
            Hessian that contains the individual Hessians as diagonal blocks.
            Has shape ``[N, A, B, ..., N, A, B, ...]``.
        """
        N, D = input.shape[0], input.shape[1:].numel()
        hessian = zeros(N, D, N, D, device=input.device, dtype=input.dtype)

        for n in range(N):
            hessian[n, :, n, :] = individual_hessians[n].reshape(D, D)

        return hessian.reshape(*input.shape, *input.shape)

    def hessian_mat_prod(self, mat: Tensor) -> Tensor:  # noqa: D102
        self.store_forward_io()
        hmp = self.problem.derivative.make_hessian_mat_prod(
            self.problem.module, None, None
        )
        return hmp(mat)
