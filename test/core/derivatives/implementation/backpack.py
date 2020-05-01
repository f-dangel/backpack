from test.core.derivatives.implementation.base import DerivativesImplementation


class BackpackDerivatives(DerivativesImplementation):
    """Derivative implementations with BackPACK."""

    def __init__(self, problem):
        problem.extend()
        super().__init__(problem)

    def jac_mat_prod(self, mat):
        """
        Input:
            mat: Matrix with which the jacobian is multiplied.
                 shape: [V, N, C_in, H_in, W_in]
        Return:
            jmp: Jacobian matrix product obatined from backPACK

        """
        g_inp, g_out = None, None

        # forward pass to initialize buffer
        self.problem.module(self.problem.input)

        return self.problem.derivative.jac_mat_prod(
            self.problem.module, g_inp, g_out, mat
        )

    def jac_t_mat_prod(self, mat):
        g_inp, g_out = None, None

        # forward pass to initialize buffer
        self.problem.module(self.problem.input)

        return self.problem.derivative.jac_t_mat_prod(
            self.problem.module, g_inp, g_out, mat
        )

    def weight_jac_t_mat_prod(self, mat, sum_batch):
        g_inp, g_out = None, None

        # forward pass to initialize buffer
        self.problem.module(self.problem.input)

        return self.problem.derivative.weight_jac_t_mat_prod(
            self.problem.module, g_inp, g_out, mat, sum_batch=sum_batch
        )

    def bias_jac_t_mat_prod(self, mat, sum_batch):
        g_inp, g_out = None, None

        # forward pass to initialize buffer
        self.problem.module(self.problem.input)

        return self.problem.derivative.bias_jac_t_mat_prod(
            self.problem.module, g_inp, g_out, mat, sum_batch=sum_batch
        )

    def weight_jac_mat_prod(self, mat):
        g_inp, g_out = None, None

        # forward pass to initialize buffer
        self.problem.module(self.problem.input)

        return self.problem.derivative.weight_jac_mat_prod(
            self.problem.module, g_inp, g_out, mat
        )

    def bias_jac_mat_prod(self, mat):
        g_inp, g_out = None, None

        # forward pass to initialize buffer
        self.problem.module(self.problem.input)

        return self.problem.derivative.bias_jac_mat_prod(
            self.problem.module, g_inp, g_out, mat
        )
