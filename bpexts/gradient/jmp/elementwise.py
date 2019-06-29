"""Jacobian-matrix product for elementwise functions."""

from torch import gt
from ...utils import einsum


def jac_mat_prod(module, grad_input, grad_output, mat, df):
    return einsum('bi,bic->bic', (df, mat))
