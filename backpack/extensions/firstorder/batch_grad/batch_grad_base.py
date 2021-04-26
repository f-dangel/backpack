from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class BatchGradBase(FirstOrderModuleExtension):
    def __init__(self, derivatives, params=None):
        self.derivatives = derivatives
        super().__init__(params=params)

    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        return self.derivatives.bias_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        return self.derivatives.weight_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )

    def bias_ih_l0(self, ext, module, g_inp, g_out, bpQuantities):
        return self.derivatives.bias_ih_l0_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )

    def bias_hh_l0(self, ext, module, g_inp, g_out, bpQuantities):
        return self.derivatives.bias_hh_l0_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )

    def weight_ih_l0(self, ext, module, g_inp, g_out, bpQuantities):
        return self.derivatives.weight_ih_l0_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )

    def weight_hh_l0(self, ext, module, g_inp, g_out, bpQuantities):
        return self.derivatives.weight_hh_l0_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )
