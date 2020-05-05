from test.core.derivatives.implementation.base import DerivativesImplementation

import torch

from backpack.hessianfree.hvp import hessian_vector_product
from backpack.hessianfree.lop import transposed_jacobian_vector_product
from backpack.hessianfree.rop import jacobian_vector_product


class AutogradDerivatives(DerivativesImplementation):
    """Derivative implementations with autograd."""

    def jac_vec_prod(self, vec):
        input, output, _ = self.problem.forward_pass(input_requires_grad=True)
        return jacobian_vector_product(output, input, vec)[0]

    def jac_mat_prod(self, mat):
        V = mat.shape[0]

        vecs = [mat[v] for v in range(V)]
        jac_vec_prods = [self.jac_vec_prod(vec) for vec in vecs]

        return torch.stack(jac_vec_prods)

    def jac_t_vec_prod(self, vec):
        input, output, _ = self.problem.forward_pass(input_requires_grad=True)
        return transposed_jacobian_vector_product(output, input, vec)[0]

    def jac_t_mat_prod(self, mat):
        V = mat.shape[0]

        vecs = [mat[v] for v in range(V)]
        jac_t_vec_prods = [self.jac_t_vec_prod(vec) for vec in vecs]

        return torch.stack(jac_t_vec_prods)

    def weight_jac_t_mat_prod(self, mat, sum_batch):
        return self.param_jac_t_mat_prod("weight", mat, sum_batch)

    def bias_jac_t_mat_prod(self, mat, sum_batch):
        return self.param_jac_t_mat_prod("bias", mat, sum_batch)

    def param_jac_t_vec_prod(self, name, vec, sum_batch):
        input, output, named_params = self.problem.forward_pass()
        param = named_params[name]

        if sum_batch:
            return transposed_jacobian_vector_product(output, param, vec)[0]
        else:
            N = input.shape[0]

            sample_outputs = [output[n] for n in range(N)]
            sample_vecs = [vec[n] for n in range(N)]

            jac_t_sample_prods = [
                transposed_jacobian_vector_product(n_out, param, n_vec)[0]
                for n_out, n_vec in zip(sample_outputs, sample_vecs)
            ]

            return torch.stack(jac_t_sample_prods)

    def param_jac_t_mat_prod(self, name, mat, sum_batch):
        V = mat.shape[0]

        vecs = [mat[v] for v in range(V)]
        jac_t_vec_prods = [
            self.param_jac_t_vec_prod(name, vec, sum_batch) for vec in vecs
        ]

        return torch.stack(jac_t_vec_prods)

    def weight_jac_mat_prod(self, mat):
        return self.param_jac_mat_prod("weight", mat)

    def bias_jac_mat_prod(self, mat):
        return self.param_jac_mat_prod("bias", mat)

    def param_jac_vec_prod(self, name, vec):
        input, output, named_params = self.problem.forward_pass()
        param = named_params[name]

        return jacobian_vector_product(output, param, vec)[0]

    def param_jac_mat_prod(self, name, mat):
        V = mat.shape[0]

        vecs = [mat[v] for v in range(V)]
        jac_vec_prods = [self.param_jac_vec_prod(name, vec) for vec in vecs]

        return torch.stack(jac_vec_prods)

    def hessian(self, loss, x):
        """Return the Hessian matrix of `loss` w.r.t. `x`.

        Arguments:
            loss (torch.Tensor): A scalar-valued tensor.
            x (torch.Tensor): Tensor used in the computation graph of `loss`.
        Shapes:
            loss: `[1,]`
            x: `[A, B, C, ...]`
        Returns:
            torch.Tensor: Hessian tensor of `loss` w.r.t. `x`. The Hessian has shape
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

    def input_hessian(self):
        """Compute the Hessian of the module output w.r.t. the input."""
        input, output, _ = self.problem.forward_pass(input_requires_grad=True)
        return self.hessian(output, input)

    def sum_hessian(self):
        """Compute the Hessian of a loss module w.r.t. its input."""
        hessian = self.input_hessian()

        return self._sum_hessian_blocks(hessian)

    def _sum_hessian_blocks(self, hessian):
        """Sum second derivatives over the batch dimension.

        Assert second derivative w.r.t. different samples is zero.
        """
        input = self.problem.input
        num_axes = len(input.shape)

        if num_axes != 2:
            raise ValueError("Only 2D inputs are currently supported.")

        N = input.shape[0]
        num_features = input.numel() // N

        sum_hessian = torch.zeros(num_features, num_features)

        hessian_different_samples = torch.zeros(num_features, num_features)
        for n_1 in range(N):
            for n_2 in range(N):
                block = hessian[n_1, :, n_2, :]

                if n_1 == n_2:
                    sum_hessian += block

                else:
                    assert torch.allclose(block, hessian_different_samples)

        return sum_hessian
