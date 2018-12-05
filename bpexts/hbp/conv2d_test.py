"""Test Hessian backpropagation of conv2d layer.

The example is taken from
    Chellapilla: High Performance Convolutional Neural Networks
    for Document Processing (2007).
"""

import torch
from torch import (Tensor, tensor, eye, randn, einsum)
from torch.nn import (Conv2d, functional)
from .conv2d import HBPConv2d
from .loss import batch_summed_hessian
from ..hessian import exact
from ..utils import torch_allclose

torch.set_printoptions(linewidth=300, threshold=10000)

# convolution parameters
in_channels = 3
out_channels = 2
kernel_size = (2, 2)
stride = (1, 1)
padding = (0, 0)
dilation = (1, 1)
bias = True

# predefined kernel matrix
kernel11 = [[1, 1],
            [2, 2]]
kernel12 = [[1, 1],
            [1, 1]]
kernel13 = [[0, 1],
            [1, 0]]
kernel21 = [[1, 0],
            [0, 1]]
kernel22 = [[2, 1],
            [2, 1]]
kernel23 = [[1, 2],
            [2, 0]]
kernel = Tensor([[kernel11, kernel12, kernel13],
                 [kernel21, kernel22, kernel23]]).float()

# bias term
b = Tensor([5, 10]).float()

# input (1 sample)
in_feature1 = [[1, 2, 0],
               [1, 1, 3],
               [0, 2, 2]]
in_feature2 = [[0, 2, 1],
               [0, 3, 2],
               [1, 1, 0]]
in_feature3 = [[1, 2, 1],
               [0, 1, 3],
               [3, 3, 2]]
in1 = Tensor([[in_feature1,
               in_feature2,
               in_feature3]]).float()
result1 = [[19, 25],
           [20, 29]]
result2 = [[22, 34],
           [27, 36]]
out1 = Tensor([[result1, result2]]).float()


def torch_example_layer():
    """Return example layer as PyTorch.nn Module."""
    conv2d = Conv2d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=bias)
    conv2d.weight.data = kernel
    conv2d.bias.data = b
    return conv2d


def example_layer():
    """Return example layer as HBP Module."""
    hbp_conv2d = HBPConv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           bias=bias)
    hbp_conv2d.weight.data = kernel
    hbp_conv2d.bias.data = b
    return hbp_conv2d


def example_loss(tensor):
    """Test loss function. Sum over squared entries."""
    return (tensor**2).contiguous().view(-1).sum()


def example_input():
    """Return example input."""
    return tensor(in1, requires_grad=True)


def test_forward():
    """Compare forward of torch.nn.Conv2d and HBPConv2d.

    Handles only single instance batch.
    """
    out_conv2d = torch_example_layer()(in1)
    assert torch_allclose(out_conv2d, out1)
    out_hbp_conv2d = example_layer()(in1)
    assert torch_allclose(out_hbp_conv2d, out1)


def layer_with_input_output_and_loss():
    """Return layer with input, output and loss."""
    layer = example_layer()
    x = example_input()
    out = layer(x)
    loss = example_loss(out)
    return layer, x, out, loss


def layer_input_hessian():
    """The Hessian with respect to the inputs by brute force."""
    layer, x, out, loss = layer_with_input_output_and_loss()
    input_hessian = exact.exact_hessian(loss, [x])
    print(input_hessian)
    return input_hessian


def test_input_hessian():
    """Check Hessian w.r.t. layer input."""
    layer, x, out, loss = layer_with_input_output_and_loss()
    # Hessian of loss function w.r.t layer output
    output_hessian = batch_summed_hessian(loss, out)
    loss_hessian = 2 * eye(8)
    assert torch_allclose(loss_hessian, output_hessian)
    # Hessian with respect to layer inputs
    h_in_result = layer_input_hessian()
    # call hessian backward
    loss.backward()
    input_hessian = layer.backward_hessian(loss_hessian)
    assert torch_allclose(input_hessian, h_in_result)


def example_bias_hessian():
    """Compute the bias Hessian via brute force."""
    layer, x, out, loss = layer_with_input_output_and_loss()
    bias_hessian = exact.exact_hessian(loss, [layer.bias])
    result = tensor([[8, 0],
                     [0, 8]]).float()
    assert torch_allclose(bias_hessian, result)
    return bias_hessian


def test_bias_hessian(random_vp=10):
    """Check correct backpropagation of bias Hessian and HVP."""
    layer, x, out, loss = layer_with_input_output_and_loss()
    # Hessian of loss function w.r.t layer output
    output_hessian = batch_summed_hessian(loss, out)
    loss.backward()
    layer.backward_hessian(output_hessian, compute_input_hessian=False)
    b_hessian = example_bias_hessian()
    assert torch_allclose(layer.bias.hessian, b_hessian)
    # check Hessian-vector product
    for _ in range(random_vp):
        v = randn(2)
        vp = layer.bias.hvp(v)
        result = b_hessian.matmul(v)
        assert torch_allclose(vp, result, atol=1E-5)


def example_weight_hessian():
    """Compute the bias Hessian via brute force."""
    layer, x, out, loss = layer_with_input_output_and_loss()
    weight_hessian = exact.exact_hessian(loss, [layer.weight])
    return weight_hessian


def test_weight_hessian(random_vp=10):
    """Check correct backpropagation of weight Hessian and HVP."""
    layer, x, out, loss = layer_with_input_output_and_loss()
    # Hessian of loss function w.r.t layer output
    output_hessian = batch_summed_hessian(loss, out)
    loss.backward()
    layer.backward_hessian(output_hessian, compute_input_hessian=False)
    w_hessian = example_weight_hessian()
    assert torch_allclose(layer.weight.hessian(), w_hessian)
    # check Hessian-vector product
    for _ in range(random_vp):
        v = randn(24)
        vp = layer.weight.hvp(v)
        result = w_hessian.matmul(v)
        assert torch_allclose(vp, result, atol=1E-5)
