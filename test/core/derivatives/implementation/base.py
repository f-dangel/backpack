class DerivativesImplementation:
    """Base class for autograd and BackPACK implementations."""

    def __init__(self, problem):
        self.problem = problem

    def jac_mat_prod(self, mat):
        raise NotImplementedError

    def jac_t_mat_prod(self, mat):
        raise NotImplementedError

    def weight_jac_t_mat_prod(self, mat, sum_batch):
        raise NotImplementedError

    def bias_jac_t_mat_prod(self, mat, sum_batch):
        raise NotImplementedError

    def weight_jac_mat_prod(self, mat):
        raise NotImplementedError

    def bias_jac_mat_prod(self, mat):
        raise NotImplementedError
