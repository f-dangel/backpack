from backpack.core.derivatives.elementwise import ElementwiseDerivatives


class SigmoidDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self):
        """`σ''(x) ≠ 0`."""
        return False

    def df(self, module, g_inp, g_out):
        """First sigmoid derivative: `σ'(x) = σ(x) (1 - σ(x))`."""
        return module.output * (1.0 - module.output)

    def d2f(self, module, g_inp, g_out):
        """Second sigmoid derivative: `σ''(x) = σ(x) (1 - σ(x)) (1 - 2 σ(x))`."""
        return module.output * (1 - module.output) * (1 - 2 * module.output)
