"""Test multiplication with the GGN."""

from torch import zeros, zeros_like
from torch.autograd import grad
from torch.nn import Linear

from backpack.hessianfree.ggnvp import ggn_vector_product


def test_ggnvp_no_explicit_dependency():
    """Test GGN-vector-product when the graph is independent of a parameter."""
    x = zeros(1, requires_grad=True)
    f = Linear(1, 1)

    y = f(x)
    # does not depend on the linear layer's bias
    (dy_dx,) = grad(y, x, create_graph=True)
    loss = (dy_dx**2).sum()

    # multiply the GGN onto a vector
    v = [zeros_like(p) for p in f.parameters()]
    ggn_vector_product(loss, dy_dx, f, v)
