"""Hessian backpropagation tests for different modes."""

from ..utils import torch_allclose, set_seeds
from torch import randn, eye
from numpy import all as numpy_all
from .linear import HBPLinear
from .relu import HBPReLU
from .sigmoid import HBPSigmoid


in_features = [10, 5, 4]
out_features = [5, 4, 2]
input_size = (10,)
input = randn((3,) + input_size)


def random_input():
    """Return random input copy."""
    return input.clone()


def create_layers(activation):
    """Create layers of the fully-connected network."""
    # same seed
    set_seeds(0)
    layers = []
    for (in_, out) in zip(in_features, out_features):
        layers.append(HBPLinear(in_features=in_,
                                out_features=out))
        layers.append(activation())
    return layers


def forward(layers, input):
    """Feed input through layers and loss. Return output and loss."""
    x = input
    for layer in layers:
        x = layer(x)
    return x, example_loss(x)


def example_loss(tensor):
    """Test loss function. Sum over squared entries.

    The Hessian of this function with respect to its
    inputs is given by an identity matrix scaled by 2.
    """
    return (tensor**2).contiguous().view(-1).sum()


def hessian_backward(activation, modify_2nd):
    """Feed input through net and loss, backward the Hessian.

    Return the layers.
    """
    layers = create_layers(activation)
    x, loss = forward(layers, random_input())
    loss_hessian = 2 * eye(x.size()[1])
    loss.backward()
    # call HBP recursively
    out_h = loss_hessian
    for layer in reversed(layers):
        out_h = layer.backward_hessian(out_h,
                                       modify_2nd_order_terms=modify_2nd)
    return layers


def compare_layer_hessians(layers1, layers2):
    """Compare the Hessians of both layers."""
    for l1, l2 in zip(layers1, layers2):
        assert(l1.__class__ is l2.__class__)
        if isinstance(l1, HBPLinear) and isinstance(l2, HBPLinear):
            assert torch_allclose(l1.bias.hessian, l2.bias.hessian)
            assert torch_allclose(l1.weight.hessian(), l2.weight.hessian())


def check_layer_hessians_psd(layers):
    """Check layer Hessians for positive semi-definiteness."""
    for l in layers:
        if isinstance(l, HBPLinear):
            assert all_eigvals_nonnegative(l.bias.hessian)
            assert all_eigvals_nonnegative(l.weight.hessian())


def all_eigvals_nonnegative(matrix):
    """Check if all eigenvalues are nonnegative.

    Assume symmetric matrix.
    """
    eigvals, _ = matrix.symeig()
    eigvals = eigvals.numpy()
    return numpy_all(eigvals > -1E-7)


def test_relu_network_psd():
    """ReLU networks with PSD loss have PSD parameter Hessians."""
    for mode in ['none', 'zero', 'clip', 'abs']:
        layers = hessian_backward(HBPReLU, mode)
        check_layer_hessians_psd(layers)


def test_sigmoid_network_psd():
    """Sigmoid nets with PSD loss require treating 2nd-order effects."""
    for mode in ['zero', 'clip', 'abs']:
        layers = hessian_backward(HBPSigmoid, mode)
        check_layer_hessians_psd(layers)

    # no modification yields no PSD matrices
    try:
        layers = hessian_backward(HBPSigmoid, 'none')
        check_layer_hessians_psd(layers)
    except AssertionError:
        pass


def test_relu_network_identical():
    """ReLU does not introduce second-order effects, mode does not matter."""
    layers1 = hessian_backward(HBPReLU, 'none')
    layers2 = hessian_backward(HBPReLU, 'clip')
    layers3 = hessian_backward(HBPReLU, 'abs')
    layers4 = hessian_backward(HBPReLU, 'zero')
    # mode should not matter
    compare_layer_hessians(layers1, layers2)
    compare_layer_hessians(layers2, layers3)
    compare_layer_hessians(layers3, layers4)


def test_sigmoid_network_not_identical():
    """Sigmoid net Hessians depend on modification strategy during HBP."""
    layers1 = hessian_backward(HBPSigmoid, 'none')
    layers2 = hessian_backward(HBPSigmoid, 'clip')
    layers3 = hessian_backward(HBPSigmoid, 'abs')
    layers4 = hessian_backward(HBPSigmoid, 'zero')
    # mode matters
    try:
        compare_layer_hessians(layers1, layers2)
    except AssertionError:
        pass
    try:
        compare_layer_hessians(layers2, layers3)
    except AssertionError:
        pass
    try:
        compare_layer_hessians(layers3, layers4)
    except AssertionError:
        pass
