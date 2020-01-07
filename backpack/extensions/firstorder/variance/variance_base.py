from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class VarianceBaseModule(FirstOrderModuleExtension):
    def __init__(self, params, grad_extension, sgs_extension):
        super().__init__(params=params)
        self.grad_ext = grad_extension
        self.sgs_ext = sgs_extension

    @staticmethod
    def variance_from(grad, sgs, N):
        avgg_squared = (grad / N) ** 2
        avg_gsquared = sgs / N
        return avg_gsquared - avgg_squared

    def bias(self, ext, module, g_inp, g_out, backproped):
        N = g_out[0].shape[0]
        return self.variance_from(
            self.grad_ext.bias(ext, module, g_inp, g_out, backproped),
            self.sgs_ext.bias(ext, module, g_inp, g_out, backproped),
            N,
        )

    def weight(self, ext, module, g_inp, g_out, backproped):
        N = g_out[0].shape[0]
        return self.variance_from(
            self.grad_ext.weight(ext, module, g_inp, g_out, backproped),
            self.sgs_ext.weight(ext, module, g_inp, g_out, backproped),
            N,
        )
