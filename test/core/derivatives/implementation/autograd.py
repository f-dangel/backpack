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
