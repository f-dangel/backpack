"""Derivatives computed with PyTorch's autograd."""
from test.core.derivatives.implementation.base import DerivativesImplementation
from typing import List

import torch
from torch import Tensor, stack, zeros_like

from backpack.hessianfree.hvp import hessian_vector_product
from backpack.hessianfree.lop import transposed_jacobian_vector_product
from backpack.hessianfree.rop import jacobian_vector_product


class AutogradDerivatives(DerivativesImplementation):
    """Derivative implementations with autograd."""

    def jac_vec_prod(self, vec) -> Tensor:
        """Product of input-output-Jacobian and a vector.

        Args:
            vec: vector

        Returns:
            product
        """
        input, output, _ = self.problem.forward_pass(input_requires_grad=True)
        return jacobian_vector_product(output, input, vec)[0]

    def jac_mat_prod(self, mat):  # noqa: D102
        try:
            return stack([self.jac_vec_prod(vec) for vec in mat])
        except RuntimeError:
            # A RuntimeError is thrown for RNNs on CUDA,
            # because PyTorch does not support double-backwards pass for them.
            # This is the recommended workaround.
            with torch.backends.cudnn.flags(enabled=False):
                return stack([self.jac_vec_prod(vec) for vec in mat])

    def jac_t_vec_prod(self, vec):  # noqa: D102
        input, output, _ = self.problem.forward_pass(input_requires_grad=True)
        return transposed_jacobian_vector_product(output, input, vec)[0]

    def jac_t_mat_prod(self, mat):  # noqa: D102
        return stack([self.jac_t_vec_prod(vec) for vec in mat])

    def weight_jac_t_mat_prod(self, mat, sum_batch, subsampling=None):  # noqa: D102
        return self.param_jac_t_mat_prod(
            "weight", mat, sum_batch, subsampling=subsampling
        )

    def bias_jac_t_mat_prod(self, mat, sum_batch):  # noqa: D102
        return self.param_jac_t_mat_prod("bias", mat, sum_batch)

    def bias_ih_l0_jac_t_mat_prod(self, mat, sum_batch):  # noqa: D102
        return self.param_jac_t_mat_prod("bias_ih_l0", mat, sum_batch, axis_batch=1)

    def bias_hh_l0_jac_t_mat_prod(self, mat, sum_batch):  # noqa: D102
        return self.param_jac_t_mat_prod("bias_ih_l0", mat, sum_batch, axis_batch=1)

    def weight_ih_l0_jac_t_mat_prod(self, mat, sum_batch):  # noqa: D102
        return self.param_jac_t_mat_prod("weight_ih_l0", mat, sum_batch, axis_batch=1)

    def weight_hh_l0_jac_t_mat_prod(self, mat, sum_batch):  # noqa: D102
        return self.param_jac_t_mat_prod("weight_hh_l0", mat, sum_batch, axis_batch=1)

    def param_jac_t_vec_prod(
        self,
        name: str,
        vec: Tensor,
        sum_batch: bool,
        axis_batch: int = 0,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Compute the product of jac_t and the given vector.

        Args:
            name: name of parameter for derivative
            vec: vectors which to multiply
            sum_batch: whether to sum along batch axis
            axis_batch: index of batch axis. Defaults to 0.
            subsampling: Indices of active samples. Default: ``None`` (all).

        Returns:
            product of jac_t and vec
        """
        input, output, named_params = self.problem.forward_pass()
        param = named_params[name]

        samples = range(input.shape[axis_batch]) if subsampling is None else subsampling
        sample_outputs = output.split(1, dim=axis_batch)
        sample_vecs = vec.split(1, dim=axis_batch)

        jac_t_sample_prods = stack(
            [
                transposed_jacobian_vector_product(sample_outputs[n], param, vec_n)[0]
                for n, vec_n in zip(samples, sample_vecs)
            ],
        )

        if sum_batch:
            jac_t_sample_prods = jac_t_sample_prods.sum(0)

        return jac_t_sample_prods

    def param_jac_t_mat_prod(
        self,
        name: str,
        mat: Tensor,
        sum_batch: bool,
        axis_batch: int = 0,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Compute the product of jac_t and the given matrix.

        Args:
            name: name of parameter for derivative
            mat: matrix which to multiply
            sum_batch: whether to sum along batch axis
            axis_batch: Batch axis, counted without the first axis. Defaults to 0.
            subsampling: Indices of active samples. Default: ``None`` (all).

        Returns:
            product of jac_t and mat
        """
        return stack(
            [
                self.param_jac_t_vec_prod(
                    name, vec, sum_batch, axis_batch=axis_batch, subsampling=subsampling
                )
                for vec in mat
            ]
        )

    def weight_jac_mat_prod(self, mat) -> Tensor:
        """Product of jacobian and matrix.

        Args:
            mat: matrix

        Returns:
            product
        """
        return self._param_jac_mat_prod("weight", mat)

    def bias_jac_mat_prod(self, mat) -> Tensor:
        """Product of jacobian and matrix.

        Args:
            mat: matrix

        Returns:
            product
        """
        return self._param_jac_mat_prod("bias", mat)

    def _param_jac_vec_prod(self, name, vec):
        input, output, named_params = self.problem.forward_pass()
        param = named_params[name]

        return jacobian_vector_product(output, param, vec)[0]

    def _param_jac_mat_prod(self, name, mat):
        return stack([self._param_jac_vec_prod(name, vec) for vec in mat])

    def ea_jac_t_mat_jac_prod(self, mat):  # noqa: D102
        def _sample_jac_t_mat_jac_prod(sample_idx, mat):
            assert len(mat.shape) == 2

            def _sample_jac_t_mat_prod(sample_idx, mat):
                sample, output, _ = self.problem.forward_pass(
                    input_requires_grad=True, sample_idx=sample_idx
                )

                result = torch.zeros(sample.numel(), mat.size(1), device=sample.device)

                for col in range(mat.size(1)):
                    column = mat[:, col].reshape(output.shape)
                    result[:, col] = transposed_jacobian_vector_product(
                        [output], [sample], [column], retain_graph=True
                    )[0].reshape(-1)

                return result

            jac_t_mat = _sample_jac_t_mat_prod(sample_idx, mat)
            mat_t_jac = jac_t_mat.t()
            jac_t_mat_t_jac = _sample_jac_t_mat_prod(sample_idx, mat_t_jac)
            jac_t_mat_jac = jac_t_mat_t_jac.t()

            return jac_t_mat_jac

        N = self.problem.input.shape[0]
        input_features = self.problem.input.shape.numel() // N

        result = torch.zeros(input_features, input_features).to(self.problem.device)

        for n in range(N):
            result += _sample_jac_t_mat_jac_prod(n, mat)

        return result / N

    def _hessian(self, loss: Tensor, x: Tensor) -> Tensor:
        """Return the Hessian matrix of a scalar `loss` w.r.t. a tensor `x`.

        Args:
            loss: A scalar-valued tensor.
            x: Tensor used in the computation graph of `loss`.

        Shapes:
            loss: `[1,]`
            x: `[A, B, C, ...]`

        Returns:
            Hessian tensor of `loss` w.r.t. `x`. The Hessian has shape
                `[A, B, C, ..., A, B, C, ...]`.
        """
        assert loss.numel() == 1

        vectorized_shape = (x.numel(), x.numel())
        final_shape = (*x.shape, *x.shape)

        hessian_vec_x = torch.zeros(vectorized_shape).to(loss.device)

        num_cols = hessian_vec_x.shape[1]
        for column_idx in range(num_cols):
            unit = torch.zeros(num_cols).to(loss.device)
            unit[column_idx] = 1.0

            unit = unit.view_as(x)
            column = hessian_vector_product(loss, [x], [unit])[0].reshape(-1)

            hessian_vec_x[:, column_idx] = column

        return hessian_vec_x.reshape(final_shape)

    def _elementwise_hessian(self, tensor: Tensor, x: Tensor) -> Tensor:
        """Computes the Hessian of each element in `tensor` w.r.t `x`.

        If ``tensor`` is linear in ``x``, autograd raises a ``RuntimeError``.
        If ``tensor`` does not depend on ``x``, autograd raises an ``AttributeError``.
        In both cases, a Hessian of zeros is created manually and returned.

        Given a `tensor` of shape `[A, B, C]` and another tensor `x` with shape `[D, E]`
        used in the computation of `tensor`, the generalized Hessian has shape
        [A, B, C, D, E, D, E]. Let `hessian` denote this generalized Hessian. Then,
        `hessian[a, b, c]` contains the Hessian of the scalar entry `tensor[a, b, c]`
        w.r.t. `x[a, b, c]`.

        If ``tensor`` is linear in ``x``, autograd raises a ``RuntimeError``.
        If ``tensor`` does not depend on ``x``, autograd raises an ``AttributeError``.
        In both cases, a Hessian of zeros is created manually and returned.

        Arguments:
            tensor: An arbitrary tensor.
            x: Tensor used in the computation graph of `tensor`.

        Yields:
            Hessians in the order of elements in the flattened tensor.
        """
        for t in tensor.flatten():
            try:
                yield self._hessian(t, x)
            except (RuntimeError, AttributeError):
                yield torch.zeros(*x.shape, *x.shape, device=x.device, dtype=x.dtype)

    def hessian_is_zero(self) -> bool:  # noqa: D102
        input, output, _ = self.problem.forward_pass(input_requires_grad=True)

        zero = None
        for hessian in self._elementwise_hessian(output, input):
            if zero is None:
                zero = zeros_like(hessian)

            if not torch.allclose(hessian, zero):
                return False

        return True

    def input_hessian(self) -> Tensor:
        """Compute the Hessian of the module output w.r.t. the input.

        Returns:
            hessian
        """
        input, output, _ = self.problem.forward_pass(input_requires_grad=True)
        return self._hessian(output, input)

    def sum_hessian(self) -> Tensor:
        """Compute the Hessian of a loss module w.r.t. its input.

        Returns:
            hessian
        """
        hessian = self.input_hessian()

        return self._sum_hessian_blocks(hessian)

    def _sum_hessian_blocks(self, hessian: Tensor) -> Tensor:
        """Sum second derivatives over the batch dimension.

        Assert second derivative w.r.t. different samples is zero.

        Args:
            hessian: .

        Returns:
            sum of hessians

        Raises:
            ValueError: if input is not 2d
        """
        input = self.problem.input
        num_axes = len(input.shape)

        if num_axes != 2:
            raise ValueError("Only 2D inputs are currently supported.")

        N = input.shape[0]
        num_features = input.numel() // N

        sum_hessian = torch.zeros(num_features, num_features, device=input.device)

        hessian_different_samples = torch.zeros(
            num_features, num_features, device=input.device
        )
        for n_1 in range(N):
            for n_2 in range(N):
                block = hessian[n_1, :, n_2, :]

                if n_1 == n_2:
                    sum_hessian += block

                else:
                    assert torch.allclose(block, hessian_different_samples)

        return sum_hessian
