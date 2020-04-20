from torch import einsum

from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class BatchL2Linear(FirstOrderModuleExtension):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        C_axis = 1
        return (g_out[0] ** 2).sum(C_axis)

    def weight(self, ext, module, g_inp, g_out, backproped):
        return einsum("ni,nj->n", (g_out[0] ** 2, module.input0 ** 2))
