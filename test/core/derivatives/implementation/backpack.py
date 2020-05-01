from test.core.derivatives.implementation.base import DerivativesImplementation


class BackpackDerivatives(DerivativesImplementation):
    """Derivative implementations with BackPACK."""

    def __init__(self, problem):
        problem.extend()
        super().__init__(problem)

    def store_forward_io(self):
        self.problem.forward_pass()

    def jac_mat_prod(self, mat):
        """
        Input:
            mat: Matrix with which the jacobian is multiplied.
                 shape: [V, N, C_in, H_in, W_in]
        Return:
            jmp: Jacobian matrix product obatined from backPACK

        """
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
