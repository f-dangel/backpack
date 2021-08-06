"""Contains derivative calculation with BackPACK."""
from itertools import product as itertools_product
from test.core.derivatives.implementation.base import DerivativesImplementation
from test.utils import chunk_sizes
from typing import Iterable, List, Tuple

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
            Hessian with respect to the input.
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

    def _sample_hessians_from_sqrt(self, sqrt):
        """Convert individual matrix square root into individual full matrix.

        Args:
            sqrt: individual square root of hessian

        Returns:
            individual full matrix

        Raises:
            ValueError: if input is not 2d
        """
        # TODO improve readability
        if sqrt.dim() >= 3:
            return einsum("vni..., vnj...->n...ij", sqrt, sqrt)
        else:
            raise ValueError("Input must have at least 2 dimensions.")

    def _embed_sample_hessians(
        self, individual_hessians: Tensor, input: Tensor
    ) -> Tensor:
        """Embed Hessians w.r.t. individual samples into Hessian w.r.t. all samples.

        Args:
            individual_hessians: Hessians w.r.t. individual samples in the input.
            input: Inputs for the individual Hessians.

        Returns:
            Hessian that contains the individual Hessians as diagonal blocks.

        Raises:
            ValueError: if input is not 2d
        """
        hessian_shape = (*input.shape, *input.shape)
        hessian = zeros(hessian_shape, device=input.device, dtype=input.dtype)

        if input.dim() >= 2:
            ranges: Tuple[Iterable, ...] = tuple(
                [range(input.shape[2 + i]) for i in range(input.dim() - 2)]
            )
            for index_n in range(input.shape[0]):
                index_additional_axes: Tuple[int]
                for index_additional_axes in itertools_product(*ranges):
                    selection = ((index_n, slice(None)) + index_additional_axes) * 2
                    hessian[selection] = individual_hessians[
                        (index_n,) + index_additional_axes
                    ]
        else:
            raise ValueError("Input must have at least 2 dimensions.")
        return hessian
