"""Test of exact Jacobian computation."""

from torch import tensor
from torch.nn import Linear
from .jacobian import exact_jacobian
from ..utils import torch_allclose


def test_exact_jacobian_simple():
    """Test exact Jacobian computation for a simple example."""
    w = tensor([[1, 2, 3],
                [4, 5, 6]]).float()
    input = tensor([[7, 8, 9],
                    [10, 11, 12]], requires_grad=True).float()
    f = input.matmul(w.transpose(0, 1))
    result = tensor([[1, 2, 3, 0, 0, 0],
                     [4, 5, 6, 0, 0, 0],
                     [0, 0, 0, 1, 2, 3],
                     [0, 0, 0, 4, 5, 6]])
    jacobian = exact_jacobian(f, [input], show_progress=True)
    print(result)
    print(jacobian)
    assert torch_allclose(result, jacobian)


def test_exact_jacobian_linear_layer():
    """Test exact Jacobian computation for a linear layer."""
    # set up linear layer
    layer = Linear(in_features=3, out_features=2, bias=True)
    w = tensor([[1, 2, 3],
                [4, 5, 6]], requires_grad=True).float()
    b = tensor([7, 8], requires_grad=True).float()
    layer.weight.data = w
    layer.bias.data = b
    # compute Jacobian of output
    input = tensor([[7, 8, 9],
                    [10, 11, 12]], requires_grad=True).float()
    f = layer(input)
    # compare results
    result = tensor([[7, 8, 9, 0, 0, 0, 1, 0],
                     [0, 0, 0, 7, 8, 9, 0, 1],
                     [10, 11, 12, 0, 0, 0, 1, 0],
                     [0, 0, 0, 10, 11, 12, 0, 1]]).float()
    jacobian = exact_jacobian(f, layer.parameters(),
                              show_progress=True)
    assert torch_allclose(result, jacobian)
