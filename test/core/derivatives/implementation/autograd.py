from test.core.derivatives.implementation.base import DerivativesImplementation

import torch
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
