from torch import einsum

from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class BatchL2Linear(FirstOrderModuleExtension):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        add_axes = list(range(1, g_out[0].dim() - 1))

        if add_axes:
            grad_batch = g_out[0].sum(add_axes)
        else:
            grad_batch = g_out[0]

        C_axis = 1

        return (grad_batch ** 2).sum(C_axis)

    def weight(self, ext, module, g_inp, g_out, backproped):
        add_axes = list(range(1, g_out[0].dim() - 1))

        if add_axes:
            # TODO Find out if einsum contraction is efficient. Otherwise it might be
            # cheaper to compute individual gradients, then take their norm
            dE_dY = g_out[0].flatten(start_dim=1, end_dim=-2)
            X = module.input0.flatten(start_dim=1, end_dim=-2)
            return einsum("nmi,nmj,nki,nkj->n", (dE_dY, X, dE_dY, X))

        else:
            return einsum("ni,nj->n", g_out[0] ** 2, module.input0 ** 2)
