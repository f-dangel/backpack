import torch


def L_op(ys, xs, ws):
    vJ = torch.autograd.grad(
        ys,
        xs,
        grad_outputs=ws,
        create_graph=True,
        retain_graph=True,
        allow_unused=True)
    return tuple([j.detach() for j in vJ])


def transposed_jacobian_vector_product(f, x, v):
    """Multiply a vector by the Jacobian.

    Corresponds to the application of the L-operator."""
    return L_op(f, x, v)
