from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class BatchGradBase(FirstOrderModuleExtension):

    def __init__(self, derivatives, params=None):
        self.derivatives = derivatives
        super().__init__(params=params)

    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        batch = g_out[0].shape[0]
        grad_out_vec = g_out[0].contiguous().view(batch, -1)

        bias_grad = self.derivatives.bias_jac_t_mat_prod(
            module, g_inp, g_out, grad_out_vec, sum_batch=False
        )

        return bias_grad.view((batch,) + module.bias.shape)

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        batch = g_out[0].shape[0]
        grad_out_vec = g_out[0].contiguous().view(batch, -1)

        weight_grad = self.derivatives.weight_jac_t_mat_prod(
            module, g_inp, g_out, grad_out_vec, sum_batch=False
        )

        return weight_grad.view((batch,) + module.weight.shape)
