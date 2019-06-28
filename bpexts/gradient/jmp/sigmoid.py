from torch import einsum


def jac_mat_prod(module, grad_input, grad_output, mat):
    d_sigma = module.output * (1. - module.output)
    jmp = einsum('bi,bic->bic', (d_sigma, mat))
    return jmp
