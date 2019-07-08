class BaseDerivatives():

    MC_SAMPLES = 1

    def jac_t_mat_prod(self, module, grad_input, grad_output, mat):
        raise NotImplementedError

    def hessian_is_zero(self):
        raise NotImplementedError

    def hessian_is_diagonal(self):
        raise NotImplementedError

    def hessian_diagonal(self):
        raise NotImplementedError
