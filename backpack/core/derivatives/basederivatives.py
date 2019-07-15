class BaseDerivatives():

    MC_SAMPLES = 1

    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        raise NotImplementedError

    def jac_t_mat_prod(self, module, grad_input, grad_output, mat):
        raise NotImplementedError

    def ea_jac_t_mat_jac_prod(self, module, grad_input, grad_output, mat):
        raise NotImplementedError

    def hessian_is_zero(self):
        raise NotImplementedError

    def hessian_is_diagonal(self):
        raise NotImplementedError

    def hessian_diagonal(self):
        raise NotImplementedError

    def hessian_is_psd(self):
        raise NotImplementedError

    @staticmethod
    def batch_flat(tensor):
        batch = tensor.size(0)
        return batch, tensor.view(batch, -1)
