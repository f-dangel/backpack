class BaseDerivatives():

    MC_SAMPLES = 1

    def jac_mat_prod(self, module, g_inp, g_out, mat):
        raise NotImplementedError

    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        raise NotImplementedError

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        raise NotImplementedError

    def hessian_is_zero(self):
        raise NotImplementedError

    def hessian_is_diagonal(self):
        raise NotImplementedError

    def hessian_diagonal(self):
        raise NotImplementedError

    def hessian_is_psd(self):
        raise NotImplementedError

    def batch_flat(self, tensor):
        batch = tensor.size(0)
        # TODO: Removing the clone().detach() will destroy the computation graph
        # Tests will fail
        return batch, tensor.clone().detach().view(batch, -1)

    def get_batch(self, module):
        return self.get_input(module).size(0)

    def get_input(self, module):
        return module.input0

    def get_output(self, module):
        return module.output
