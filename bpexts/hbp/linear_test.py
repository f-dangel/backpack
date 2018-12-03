"""Test Hessian backpropagation of linear layer."""

from torch import (tensor, randn)
from .linear import HBPLinear
from .loss import batch_summed_hessian
from ..utils import torch_allclose


def example_layer():
    """Return example linear layer."""
    layer = HBPLinear(in_features=3, out_features=2)
    w = tensor([[1, 2, 3],
                [4, 5, 6]], requires_grad=True).float()
    b = tensor([7, 8], requires_grad=True).float()
    layer.weight.data = w
    layer.bias.data = b
    return layer


def example_input():
    """Example input for the linear layer."""
    x = tensor([[1, 3, 5]], requires_grad=True).float()
    return x


def example_loss(tensor):
    """Sum all square entries of a tensor."""
    return (tensor**2).view(-1).sum(0)


def test_mean_input_hook():
    """Check storing of mean_input hook."""
    layer = example_layer()
    x = tensor([[2, 4, 6],
                [8, 10, 12],
                [14, 16, 18]]).float()
    mean_x = tensor([[8],
                     [10],
                     [12]])
    # forward, calls hook
    layer(x)
    assert torch_allclose(layer.mean_input, mean_x)


def test_input_hessian():
    """Return layer after backward_hessian, check input Hessian."""
    layer, x = example_layer(), example_input()
    out = layer(x)
    loss = example_loss(out)
    # Hessian of loss function w.r.t layer output
    output_hessian = batch_summed_hessian(loss, out)
    loss_hessian = tensor([[2, 0],
                           [0, 2]]).float()
    assert torch_allclose(loss_hessian, output_hessian)
    # Hessian with respect to layer inputs
    # call hessian backward
    input_hessian = layer.backward_hessian(loss_hessian)
    # result: W^T * output_hessian * W
    h_in_result = tensor([[34, 44, 54],
                          [44, 58, 72],
                          [54, 72, 90]]).float()
    assert torch_allclose(input_hessian, h_in_result)
    return layer


def test_bias_hessian(random_vp=10):
    """Check correct backpropagation of bias Hessian and HVP."""
    layer = test_input_hessian()
    # Hessian with respect to layer bias
    bias_hessian = tensor([[2, 0],
                           [0, 2]]).float()
    assert torch_allclose(layer.bias.hessian, bias_hessian)
    # check Hessian-vector product
    for _ in range(random_vp):
        v = randn(2)
        vp = layer.bias.hvp(v)
        result = bias_hessian.matmul(v)
        assert torch_allclose(vp, result)


def test_weight_hessian(random_vp=10):
    """Check correct weight Hessian/HVP backpropagation."""
    layer = test_input_hessian()
    # Hessian with respect to layer weights
    # x * x^T \otimes output_hessian
    weight_hessian = tensor([[2, 6, 10, 0, 0, 0],
                             [6, 18, 30, 0, 0, 0],
                             [10, 30, 50, 0, 0, 0],
                             [0, 0, 0, 2, 6, 10],
                             [0, 0, 0, 6, 18, 30],
                             [0, 0, 0, 10, 30, 50]]).float()
    print(layer.weight.hessian())
    assert torch_allclose(layer.weight.hessian(), weight_hessian)
    # check Hessian-vector product
    for _ in range(random_vp):
        v = randn(6)
        vp = layer.weight.hvp(v)
        result = weight_hessian.matmul(v)
        assert torch_allclose(vp, result)
