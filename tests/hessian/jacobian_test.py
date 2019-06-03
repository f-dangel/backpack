"""Test of exact Jacobian computation."""

import torch
from torch import (tensor, cat)
from torch.nn import Linear
from bpexts.hessian.jacobian import jacobian


def simple_example():
    """Return weight, input, function of a simple example."""
    w = tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    # 2 samples
    x = tensor([[7., 8., 9.], [10., 11., 12.]], requires_grad=True)
    f = x.matmul(w.transpose(0, 1))
    return w, x, f


def test_jacobian_simple():
    """Test Jacobian computation for a simple example."""
    w, x, f = simple_example()
    # Jacobian with respect to input
    result = tensor([[1., 2., 3., 0., 0., 0.], [4., 5., 6., 0., 0., 0.],
                     [0., 0., 0., 1., 2., 3.], [0., 0., 0., 4., 5., 6.]])
    jac = jacobian(f, x, show_progress=True)
    assert torch.allclose(result, jac)
    # Jacobian with respect to weight
    result2 = tensor([[7., 8., 9., 0., 0., 0.], [0., 0., 0., 7., 8., 9.],
                      [10., 11., 12., 0., 0., 0.], [0., 0., 0., 10., 11.,
                                                    12.]])
    jac2 = jacobian(f, w, show_progress=True)
    assert torch.allclose(result2, jac2)


def test_jacobian_batched_simple():
    """Test Jacobian computation for batched quantity, simple example."""
    w, x, f = simple_example()
    # Note how the zeros of the previous example are avoided
    result = tensor([[[1., 2., 3.], [4., 5., 6.]], [[1., 2., 3.], [4., 5.,
                                                                   6.]]])
    jac = jacobian(f, x, batched_f=True, batched_x=True, show_progress=True)
    assert torch.allclose(result, jac)


def linear_layer_example():
    """Return layer, input of a linear layer example."""
    # set up linear layer
    layer = Linear(in_features=3, out_features=2, bias=True)
    w = tensor([[1., 2, 3], [4, 5, 6]], requires_grad=True)
    b = tensor([7., 8], requires_grad=True)
    layer.weight.data = w
    layer.bias.data = b
    # two samples
    x = tensor([[7., 8, 9], [10, 11, 12]], requires_grad=True)
    return layer, x


def test_jacobian_linear_layer():
    """Test Jacobian computation for a linear layer."""
    layer, x = linear_layer_example()
    f = layer(x)
    result_w = tensor([[7., 8, 9, 0, 0, 0], [0, 0, 0, 7, 8, 9],
                       [10, 11, 12, 0, 0, 0], [0, 0, 0, 10, 11, 12]])
    jac_w = jacobian(f, layer.weight, show_progress=True)
    assert torch.allclose(result_w, jac_w)
    result_b = tensor([[1., 0], [0, 1], [1, 0], [0, 1]])
    jac_b = jacobian(f, layer.bias, show_progress=True)
    assert torch.allclose(result_b, jac_b)


def test_jacobian_batched_linear_layer():
    """Test Jacobian computation for batched quantities of a linear layer."""
    layer, x = linear_layer_example()
    f = layer(x)
    result = tensor([[[1., 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    jac = jacobian(f, x, batched_f=True, batched_x=True, show_progress=True)
    assert torch.allclose(result, jac)
