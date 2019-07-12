import torch


def R_op(ys, xs, vs, retain_graph=True, detach=True):
    if isinstance(ys, tuple):
        ws = [torch.zeros_like(y).requires_grad_(True) for y in ys]
    else:
        ws = torch.zeros_like(ys).requires_grad_(True)

    gs = torch.autograd.grad(
        ys,
        xs,
        grad_outputs=ws,
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=True)
    re = torch.autograd.grad(
        gs,
        ws,
        grad_outputs=vs,
        create_graph=True,
        retain_graph=True,
        allow_unused=True)
    if detach:
        return tuple([j.detach() for j in re])
    else:
        return re


def jacobian_vector_product(f, x, v, retain_graph=True, detach=True):
    """Multiply a vector by the Jacobian.

    Corresponds to the application of the R-operator."""
    return R_op(f, x, v, retain_graph=retain_graph, detach=detach)
