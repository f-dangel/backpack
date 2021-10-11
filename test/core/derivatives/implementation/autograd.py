"""Derivatives computed with PyTorch's autograd."""
from test.core.derivatives.implementation.base import DerivativesImplementation
from typing import List

from torch import Tensor, allclose, backends, cat, stack, zeros, zeros_like

from backpack.hessianfree.hvp import hessian_vector_product
from backpack.hessianfree.lop import transposed_jacobian_vector_product
from backpack.hessianfree.rop import jacobian_vector_product
from backpack.utils.subsampling import subsample


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
            with backends.cudnn.flags(enabled=False):
                return stack([self.jac_vec_prod(vec) for vec in mat])

    def jac_t_vec_prod(self, vec: Tensor, subsampling=None) -> Tensor:  # noqa: D102
        input, output, _ = self.problem.forward_pass(input_requires_grad=True)

        if subsampling is None:
            return transposed_jacobian_vector_product(output, input, vec)[0]
        else:
            # for each sample, multiply by full input Jacobian, slice out result:
            # ( (∂ output[n] / ∂ input)ᵀ v[n] )[n]
            batch_axis = 0
            output = subsample(output, dim=batch_axis, subsampling=subsampling)
            output = output.split(1, dim=batch_axis)
            vec = vec.split(1, dim=batch_axis)

            vjps: List[Tensor] = []
            for sample_idx, out, v in zip(subsampling, output, vec):
                vjp = transposed_jacobian_vector_product(out, input, v)[0]
                vjp = subsample(vjp, dim=batch_axis, subsampling=[sample_idx])
                vjps.append(vjp)

            return cat(vjps, dim=batch_axis)

    def jac_t_mat_prod(
        self, mat: Tensor, subsampling: List[int] = None
    ) -> Tensor:  # noqa: D102
        return stack([self.jac_t_vec_prod(vec, subsampling=subsampling) for vec in mat])

    def param_mjp(
        self,
        param_str: str,
        mat: Tensor,
        sum_batch: bool,
        subsampling: List[int] = None,
    ) -> Tensor:  # noqa: D102
        return stack(
            [
                self._param_vjp(
                    param_str,
                    vec,
                    sum_batch,
                    axis_batch=0,
                    subsampling=subsampling,
                )
                for vec in mat
            ]
        )

    def _param_vjp(
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
                    input_requires_grad=True, subsampling=[sample_idx]
                )

                result = zeros(sample.numel(), mat.size(1), device=sample.device)

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

        result = zeros(input_features, input_features).to(self.problem.device)

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

        hessian_vec_x = zeros(vectorized_shape).to(loss.device)

        num_cols = hessian_vec_x.shape[1]
        for column_idx in range(num_cols):
            unit = zeros(num_cols).to(loss.device)
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
                yield zeros(*x.shape, *x.shape, device=x.device, dtype=x.dtype)

    def hessian_is_zero(self) -> bool:  # noqa: D102
        input, output, _ = self.problem.forward_pass(input_requires_grad=True)

        zero = None
        for hessian in self._elementwise_hessian(output, input):
            if zero is None:
                zero = zeros_like(hessian)

            if not allclose(hessian, zero):
                return False

        return True

    def input_hessian(self, subsampling: List[int] = None) -> Tensor:
        """Compute the Hessian of the module output w.r.t. the input.

        Args:
            subsampling: Indices of active samples. ``None`` uses all samples.

        Returns:
            Hessian of shape ``[N, *, N, *]`` where ``N`` denotes the
            number of sub-samples, and ``*`` is the input feature shape.
        """
        input, output, _ = self.problem.forward_pass(input_requires_grad=True)
        hessian = self._hessian(output, input)
        return self._subsample_input_hessian(hessian, input, subsampling=subsampling)

    @staticmethod
    def _subsample_input_hessian(
        hessian: Tensor, input: Tensor, subsampling: List[int] = None
    ) -> Tensor:
        """Slice sub-samples out of Hessian w.r.t the full input.

        If ``subsampling`` is set to ``None``, leaves the Hessian unchanged.

        Args:
            hessian: The Hessian w.r.t. the module input.
            input: Module input.
            subsampling: List of active samples. Default of ``None`` uses all samples.

        Returns:
            Sub-sampled Hessian of shape ``[N, *, N, *]`` where ``N`` denotes the
            number of sub-samples, and ``*`` is the input feature shape.
        """
        N, D_shape = input.shape[0], input.shape[1:]
        D = input.numel() // N

        subsampled_hessian = hessian.reshape(N, D, N, D)[subsampling, :, :, :][
            :, :, subsampling, :
        ]

        has_duplicates = subsampling is not None and len(set(subsampling)) != len(
            subsampling
        )
        if has_duplicates:
            # For duplicates in `subsampling`, the above slicing is not sufficient.
            # and off-diagonal blocks need to be zeroed. E.g. if subsampling is [0, 0]
            # then the sliced input Hessian has non-zero off-diagonal blocks (1, 0) and
            # (0, 1), which should be zero as the samples are considered independent.
            for idx1, sample1 in enumerate(subsampling[:-1]):
                for idx2, sample2 in enumerate(subsampling[idx1 + 1 :], start=idx1 + 1):
                    if sample1 == sample2:
                        subsampled_hessian[idx1, :, idx2, :] = 0
                        subsampled_hessian[idx2, :, idx1, :] = 0

        N_active = N if subsampling is None else len(subsampling)
        out_shape = [N_active, *D_shape, N_active, *D_shape]

        return subsampled_hessian.reshape(out_shape)

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
            hessian: Hessian of the output w.r.t. the input. Has shape ``[N, *, N, *]``
                where ``N`` is the number of active samples and ``*`` is the input's
                feature shape.

        Returns:
            Sum of Hessians w.r.t. to individual samples. Has shape ``[*, *]``.
        """
        input = self.problem.input
        N = input.shape[0]
        shape_feature = input.shape[1:]
        D = shape_feature.numel()

        hessian = hessian.reshape(N, D, N, D)
        sum_hessian = zeros(D, D, device=input.device, dtype=input.dtype)

        hessian_different_samples = zeros(D, D, device=input.device, dtype=input.dtype)
        for n_1 in range(N):
            for n_2 in range(N):
                block = hessian[n_1, :, n_2, :]
                if n_1 == n_2:
                    sum_hessian += block
                else:
                    assert allclose(block, hessian_different_samples)

        return sum_hessian.reshape(*shape_feature, *shape_feature)

    def hessian_mat_prod(self, mat: Tensor) -> Tensor:  # noqa: D102
        input, output, _ = self.problem.forward_pass(input_requires_grad=True)

        return stack([hessian_vector_product(output, [input], [vec])[0] for vec in mat])
