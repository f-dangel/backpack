from torch import einsum


def extract_weight_diagonal(module, backproped, sum_batch=True):
    if sum_batch:
        return einsum("vno,ni->oi", (backproped ** 2, module.input0 ** 2))
    else:
        return einsum("vno,ni->noi", (backproped ** 2, module.input0 ** 2))


def extract_bias_diagonal(module, backproped, sum_batch=True):
    if sum_batch:
        return einsum("vno->o", backproped ** 2)
    else:
        return einsum("vno->no", backproped ** 2)
