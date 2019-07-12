import torch


def L_op(ys, xs, ws, retain_graph=True, detach=True):
    vJ = torch.autograd.grad(
        ys,
        xs,
        grad_outputs=ws,
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=True)
    if detach:
        return tuple([j.detach() for j in vJ])
    else:
        return vJ


def transposed_jacobian_vector_product(f, x, v, retain_graph=True, detach=True):
    """Multiply a vector by the Jacobian.

    Corresponds to the application of the L-operator."""
    return L_op(f, x, v, retain_graph=retain_graph, detach=True)
