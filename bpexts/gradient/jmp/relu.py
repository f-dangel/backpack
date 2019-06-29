from torch import gt
from .elementwise import jac_mat_prod as elementwise_jac_mat_prod


def jac_mat_prod(module, grad_input, grad_output, mat):
    d_relu = gt(module.input0, 0).float()
    return elementwise_jac_mat_prod(module, grad_input, grad_output, mat,
                                    d_relu)
