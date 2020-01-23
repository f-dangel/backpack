import torch


def R_op(ys, xs, vs, retain_graph=True, detach=True):
    """
    Multiplies the vector `vs` with the Jacobian of `ys` w.r.t. `xs`.
    """

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
        allow_unused=True,
    )

    re = torch.autograd.grad(
        gs, ws, grad_outputs=vs, create_graph=True, retain_graph=True, allow_unused=True
    )

    if detach:
        return tuple(j.detach() for j in re)
    else:
        return re


def jacobian_vector_product(f, x, v, retain_graph=True, detach=True):
    """
    Multiplies the vector `v` with the Jacobian of `f` w.r.t. `x`.

    Corresponds to the application of the R-operator.

    Parameters:
    -----------
        f: torch.Tensor or [torch.Tensor]
            Outputs of the differentiated function.
        x: torch.Tensor or [torch.Tensor]
            Inputs w.r.t. which the gradient will be returned.
        v: torch.Tensor or [torch.Tensor]
            The vector to be multiplied by the Jacobian.
        retain_graph: Bool, optional
            If False, the graph used to compute the grad will be freed.
            (default: True)
        detach: Bool, optional
            If True, the Jacobian-vector product will be detached.
            (default: True)
    """
    return R_op(f, x, v, retain_graph=retain_graph, detach=detach)
