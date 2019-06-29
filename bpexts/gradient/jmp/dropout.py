from torch import eq
from ...utils import einsum


def jac_mat_prod(module, grad_input, grad_output, mat):
    scaling = 1 / (1 - module.p)
    mask = 1 - eq(grad_input, 0.).float()
    d_dropout = mask * scaling
    jmp = einsum('bo,boc->boc', d_dropout, mat)
    return jmp
