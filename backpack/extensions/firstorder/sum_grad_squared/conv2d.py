from backpack.utils.utils import einsum
from backpack.utils import conv as convUtils
from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class SGSConv2d(FirstOrderModuleExtension):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        return (g_out[0].sum(3).sum(2)**2).sum(0)

    def weight(self, ext, module, g_inp, g_out, backproped):
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, g_out[0], module)
        d1 = einsum('bml,bkl->bmk', (dE_dY, X))
        return (d1**2).sum(0).view_as(module.weight)


class SGSConv2dConcat(FirstOrderModuleExtension):
    def __init__(self):
        super().__init__(params=["weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, g_out[0], module)

        if module.has_bias():
            X = module.append_ones(X)

        d1 = einsum('bml,bkl->bmk', (dE_dY, X))
        return (d1**2).sum(0).view_as(module.weight)

