"""Test of exact Jacobian computation."""

from torch import tensor
from torch.nn import Linear
from .jacobian import (exact_jacobian,
                       exact_jacobian_batchwise)
from ..utils import torch_allclose


def simple_example():
    """Return weight, input, function of a simple example."""
    w = tensor([[1, 2, 3],
                [4, 5, 6]], requires_grad=True).float()
    # 2 samples
    input = tensor([[7, 8, 9],
                    [10, 11, 12]], requires_grad=True).float()
    f = input.matmul(w.transpose(0, 1))
    return w, input, f
 

def test_exact_jacobian_simple():
    """Test exact Jacobian computation for a simple example."""
    w, input, f = simple_example()
    # Jacobian with respect to input
    result = tensor([[1, 2, 3, 0, 0, 0],
                     [4, 5, 6, 0, 0, 0],
                     [0, 0, 0, 1, 2, 3],
                     [0, 0, 0, 4, 5, 6]]).float()
    jacobian = exact_jacobian(f, [input], show_progress=True)
    assert torch_allclose(result, jacobian)
    # Jacobian with respect to weight
    result2 = tensor([[7, 8, 9, 0, 0, 0],
                      [0, 0, 0, 7, 8, 9],
                      [10, 11, 12, 0, 0, 0],
                      [0, 0, 0, 10, 11, 12]]).float()
    jacobian2 = exact_jacobian(f, [w], show_progress=True)
    assert torch_allclose(result2, jacobian2)


def test_exact_jacobian_batchwise_simple():
    """Test batchwise exact Jacobian computation, simple examle."""
    w, input, f = simple_example()
    # Note how the zeros of the previous example are avoided
    result = tensor([[[1, 2, 3],
                      [4, 5, 6]],
                     [[1, 2, 3],
                      [4, 5, 6]]]).float()
    jacobian = exact_jacobian_batchwise(f, input, show_progress=True)
    assert torch_allclose(result, jacobian)


def linear_layer_example():
    """Return layer, input of a linear layer example."""
    # set up linear layer
    layer = Linear(in_features=3, out_features=2, bias=True)
    w = tensor([[1, 2, 3],
                [4, 5, 6]], requires_grad=True).float()
    b = tensor([7, 8], requires_grad=True).float()
    layer.weight.data = w
    layer.bias.data = b
    # two samples
    input = tensor([[7, 8, 9],
                    [10, 11, 12]], requires_grad=True).float()
    return layer, input


def test_exact_jacobian_linear_layer():
    """Test exact Jacobian computation for a linear layer."""
    layer, input = linear_layer_example()
    f = layer(input)
    result = tensor([[7, 8, 9, 0, 0, 0, 1, 0],
                     [0, 0, 0, 7, 8, 9, 0, 1],
                     [10, 11, 12, 0, 0, 0, 1, 0],
                     [0, 0, 0, 10, 11, 12, 0, 1]]).float()
    jacobian = exact_jacobian(f, layer.parameters(), show_progress=True)
    assert torch_allclose(result, jacobian)


def test_exact_jacobian_batchwise_linear_layer():
    """Test batchwise Jacobian computation for a linear layer."""
    layer, input = linear_layer_example()
    f = layer(input)
    result = tensor([[[1, 2, 3],
                      [4, 5, 6]],
                     [[1, 2, 3],
                      [4, 5, 6]]])
    jacobian = exact_jacobian_batchwise(f, input, show_progress=True)
    assert torch_allclose(result, jacobian)
