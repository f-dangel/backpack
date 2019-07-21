from backpack.utils.utils import einsum
from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class BatchL2Linear(FirstOrderModuleExtension):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        return (g_out[0] ** 2).sum(1)

    def weight(self, ext, module, g_inp, g_out, backproped):
        return einsum('bi,bj->b', (g_out[0] ** 2, module.input0 ** 2))


class BatchL2LinearConcat(FirstOrderModuleExtension):
    def __init__(self):
        super().__init__(params=["weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        input = module.homogeneous_input()
        return einsum('bi,bj->b', (g_out[0] ** 2, input ** 2))
