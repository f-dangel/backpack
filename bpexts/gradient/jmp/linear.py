from torch import einsum


def jac_mat_prod(module, grad_input, grad_output, mat):
    d_linear = module.weight
    jmp = einsum('ij,bic->bjc', (d_linear, mat))
    return jmp
