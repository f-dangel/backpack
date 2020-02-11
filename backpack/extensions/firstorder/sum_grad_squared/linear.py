from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.utils.ein import einsum


class SGSLinear(FirstOrderModuleExtension):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        N_axis = 0
        return (g_out[0] ** 2).sum(N_axis)

    def weight(self, ext, module, g_inp, g_out, backproped):
        return einsum("ni,nj->ij", (g_out[0] ** 2, module.input0 ** 2))
