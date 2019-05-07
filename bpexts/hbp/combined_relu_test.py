"""Test Hessian backpropagation of composition layer (ReLU).

This file is a copy of combined_sigmoid_test.py
"""

from torch import (tensor, randn)
from torch.nn import functional
from .linear import HBPLinear
from .relu import HBPReLU
from .combined_relu import HBPReLULinear
from .loss import batch_summed_hessian
from ..hessian.exact import exact_hessian
from ..utils import torch_allclose


def example_parameters():
    """Return example parameters w, b."""
    w = tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    b = tensor([7., 8.], requires_grad=True)
    return w, b


def example_composition_layer():
    """Return example composition layer."""
    layer = HBPReLULinear(in_features=3, out_features=2)
    w, b = example_parameters()
    layer.linear.weight.data = w
    layer.linear.bias.data = b
    return layer


def example_input():
    """Return example input."""
    return tensor([[0.1, 0.2, 0.3]], requires_grad=True).float()


def example_loss(tensor):
    """Sum all square entries of a tensor."""
    return (tensor**2).view(-1).sum(0)


def test_composition_disable_exts():
    """Test disabling of extensions works for compositions."""
    layer = example_composition_layer()
    assert bool(layer.activation.exts_hooks)
    assert bool(layer.linear.exts_hooks)
    layer.disable_exts()
    assert not bool(layer.linear.exts_hooks)
    assert not bool(layer.activation.exts_hooks)


def composition_layer_with_input_output_and_loss():
    """Return layer with inputs, output and loss."""
    layer = example_composition_layer()
    x = example_input()
    out = layer(x)
    loss = example_loss(out)
    return layer, x, out, loss


def composition_layer_input_hessian():
    """Compute the Hessian with respect to the input."""
    layer, x, out, loss = composition_layer_with_input_output_and_loss()
    input_hessian = exact_hessian(loss, [x])
    return input_hessian


def test_composition_forward():
    """Test for identical forward pass."""
    w, b = example_parameters()
    layer, x, out, _ = composition_layer_with_input_output_and_loss()
    result = functional.linear(functional.relu(x), w, bias=b)
    assert torch_allclose(out, result)


def test_input_hessian():
    """Test input Hessian of composition layer, return layer after HBP"""
    layer, x, out, loss = composition_layer_with_input_output_and_loss()
    # Hessian of loss function w.r.t layer output/input
    output_hessian = batch_summed_hessian(loss, out)
    # Hessian with respect to layer inputs
    # call hessian backward
    loss.backward()
    input_hessian = layer.backward_hessian(output_hessian)
    h_in_result = composition_layer_input_hessian()
    assert torch_allclose(input_hessian, h_in_result)
    return layer
