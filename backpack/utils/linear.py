from torch import einsum


def extract_weight_diagonal(module, backproped, sum_batch=True):
    if sum_batch:
        equation = "vno,ni->oi"
    else:
        equation = "vno,ni->noi"
    return einsum(equation, (backproped ** 2, module.input0 ** 2))


def extract_bias_diagonal(module, backproped, sum_batch=True):
    if sum_batch:
        equation = "vno->o"
    else:
        equation = "vno->no"
    return einsum(equation, backproped ** 2)
