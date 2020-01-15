from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.utils.einsum import einsum


class BatchL2Linear(FirstOrderModuleExtension):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        return (g_out[0] ** 2).sum(1)

    def weight(self, ext, module, g_inp, g_out, backproped):
        return einsum("bi,bj->b", (g_out[0] ** 2, module.input0 ** 2))
