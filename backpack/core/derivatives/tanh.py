from backpack.core.derivatives.elementwise import ElementwiseDerivatives


class TanhDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self):
        return False

    def df(self, module, g_inp, g_out):
        return 1.0 - module.output ** 2

    def d2f(self, module, g_inp, g_out):
        return -2.0 * module.output * (1.0 - module.output ** 2)
