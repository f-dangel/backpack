from torch import einsum


def extract_weight_diagonal(module, backproped):
    return einsum("vno,ni->oi", (backproped ** 2, module.input0 ** 2))


def extract_bias_diagonal(module, backproped):
    return einsum("vno->o", backproped ** 2)
