import torch


class DerivativesImplementation:
    """Base class for autograd and BackPACK implementations.

    self.input_mat: The input matrix required for testing is initialized
    """

    def __init__(self, problem):
        self.problem = problem

    def jac_mat_prod(self, mat):
        raise NotImplementedError

    def jac_t_mat_prod(self, mat):
        raise NotImplementedError
