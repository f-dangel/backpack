class BaseJacobian():

    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        raise NotImplementedError
