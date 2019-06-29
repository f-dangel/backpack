from torch import gt
from ...utils import einsum


def jac_mat_prod(module, grad_input, grad_output, mat):
    d_relu = gt(module.input0, 0).float()
    jmp = einsum('bi,bic->bic', (d_relu, mat))
    return jmp
