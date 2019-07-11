import torch

from .rop import R_op


def hessian_vector_product(f, x, v):
    """Compute Hessian-vector product Hf(x) * v."""
    df_dx = torch.autograd.grad(f, x, create_graph=True, retain_graph=True)
    Hv = R_op(df_dx, x, v)
    return tuple([j.detach() for j in Hv])
