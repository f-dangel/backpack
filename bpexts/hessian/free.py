"""Compute Jacobian-vector and Hessian-vector products using auto-diff.

Implementation by Piotr Sokol

Reference
---------
* Original snippet
    https://discuss.pytorch.org/t/adding-functionality-hessian-and-fisher-information-vector-products/23295

* Information about the L- and R-operator:
    TODO
"""

import torch


def L_op(ys, xs, ws):
    vJ = torch.autograd.grad(
        ys,
        xs,
        grad_outputs=ws,
        create_graph=True,
        retain_graph=True,
        allow_unused=True)
    print('L_op')
    print([i.size() for i in vJ])
    return tuple([j.detach() for j in vJ])


def transposed_jacobian_vector_product(f, x, v):
    """Multiply a vector by the Jacobian.

    Corresponds to the application of the L-operator."""
    return L_op(f, x, v)
