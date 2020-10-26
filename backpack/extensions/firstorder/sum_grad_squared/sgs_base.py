from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class SGSBase(FirstOrderModuleExtension):
    def __init__(self, derivatives, params=None):
        self.derivatives = derivatives
        self.N_axis = 0
        super().__init__(params=params)

    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        grad_batch = self.derivatives.bias_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )
        return (grad_batch ** 2).sum(self.N_axis)

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        grad_batch = self.derivatives.weight_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )
        return (grad_batch ** 2).sum(self.N_axis)
