import torch


def L_op(ys, xs, ws, retain_graph=True, detach=True):
    """
    Multiplies the vector `ws` with the transposed Jacobian of `ys` w.r.t. `xs`.
    """

    vJ = torch.autograd.grad(
        ys,
        xs,
        grad_outputs=ws,
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    if detach:
        return tuple(j.detach() for j in vJ)
    else:
        return vJ


def transposed_jacobian_vector_product(f, x, v, retain_graph=True, detach=True):
    """
    Multiplies the vector `v` with the (transposed) Jacobian of `f` w.r.t. `x`.

    Corresponds to the application of the L-operator.

    Parameters:
    -----------
        f: torch.Tensor or [torch.Tensor]
            Outputs of the differentiated function.
        x: torch.Tensor or [torch.Tensor]
            Inputs w.r.t. which the gradient will be returned.
        v: torch.Tensor or [torch.Tensor]
            The vector to be multiplied by the transposed Jacobian.
        retain_graph: Bool, optional
            If False, the graph used to compute the grad will be freed.
            (default: True)
        detach: Bool, optional
            If True, the transposed Jacobian-vector product will be detached.
            (default: True)
    """
    return L_op(f, x, v, retain_graph=retain_graph, detach=detach)
