from .elementwise import jac_mat_prod as elementwise_jac_mat_prod


def jac_mat_prod(module, grad_input, grad_output, mat):
    d_tanh = 1. - module.output**2
    return elementwise_jac_mat_prod(module, grad_input, grad_output, mat,
                                    d_tanh)
