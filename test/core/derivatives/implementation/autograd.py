from test.core.derivatives.implementation.base import DerivativesImplementation

import torch
from backpack.hessianfree.lop import transposed_jacobian_vector_product
from backpack.hessianfree.rop import jacobian_vector_product


class AutogradDerivatives(DerivativesImplementation):
    """Derivative implementations with autograd."""

    def jac_vec_prod(self, vec):
        """
        Input:
            input_mat: input matrix.
                       shape: [N, C, H_in, W_in]
                  mat: Matrix with which the jacobian is multiplied.
                       shape: [N, C_in, H_in, W_in]
        Returns:
             jmp_vec: Jacobian matrix product obatined from torch.autograd
        """
        input = self.problem.input
        input.requires_grad = True
        output = self.problem.module(input)

        return jacobian_vector_product(output, input, vec)[0]

    def jac_mat_prod(self, mat):
        """
        Input:
            mat: Matrix with which the jacobian is multiplied.
                 shape: [V, N, C_in, H_in, W_in]
        Return:
            jmp: Jacobian matrix product obatined from torch.autograd

        """
        V = mat.shape[0]

        jac_vec_prods = []

        for v in range(V):
            vec = mat[v]
            jac_vec_prods.append(self.jac_vec_prod(vec))

        return torch.stack(jac_vec_prods)

    def jac_t_vec_prod(self, vec):
        input = self.problem.input
        input.requires_grad = True
        output = self.problem.module(input)

        return transposed_jacobian_vector_product(output, input, vec)[0]

    def jac_t_mat_prod(self, mat):
        V = mat.shape[0]

        jac_t_vec_prods = []

        for v in range(V):
            vec = mat[v]
            jac_t_vec_prods.append(self.jac_t_vec_prod(vec))

        return torch.stack(jac_t_vec_prods)

    def weight_jac_t_mat_prod(self, mat, sum_batch):
        return self.param_jac_t_mat_prod("weight", mat, sum_batch)

    def bias_jac_t_mat_prod(self, mat, sum_batch):
        return self.param_jac_t_mat_prod("bias", mat, sum_batch)

    def param_jac_t_vec_prod(self, name, vec, sum_batch):
        input = self.problem.input
        output = self.problem.module(input)

        param = getattr(self.problem.module, name)

        if sum_batch:
            return transposed_jacobian_vector_product(output, param, vec)[0]
        else:
            N = input.shape[0]

            jac_t_sample_prods = []
            for n in range(N):
                jac_t_sample_prods.append(
                    transposed_jacobian_vector_product(output[n], param, vec[n])[0]
                )

            return torch.stack(jac_t_sample_prods)

    def param_jac_t_mat_prod(self, name, mat, sum_batch):
        V = mat.shape[0]

        jac_t_vec_prods = []

        for v in range(V):
            vec = mat[v]
            jac_t_vec_prods.append(self.param_jac_t_vec_prod(name, vec, sum_batch))

        return torch.stack(jac_t_vec_prods)

    def weight_jac_mat_prod(self, mat):
        return self.param_jac_mat_prod("weight", mat)

    def bias_jac_mat_prod(self, mat):
        return self.param_jac_mat_prod("bias", mat)

    def param_jac_vec_prod(self, name, vec):
        input = self.problem.input
        output = self.problem.module(input)
        param = getattr(self.problem.module, name)

        return jacobian_vector_product(output, param, vec)[0]

    def param_jac_mat_prod(self, name, mat):
        V = mat.shape[0]

        jac_vec_prods = []

        for v in range(V):
            vec = mat[v]
            jac_vec_prods.append(self.param_jac_vec_prod(name, vec))

        return torch.stack(jac_vec_prods)
