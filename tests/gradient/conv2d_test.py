"""
Test batch gradient computation of conv2d layer.

The example is taken from
    Chellapilla: High Performance Convolutional Neural Networks
    for Document Processing (2007).
"""

from torch import (Tensor, randn)
from torch.nn import Conv2d
from random import (randint, choice)
from bpexts.gradient.conv2d import Conv2d as G_Conv2d
import bpexts.gradient.config as config
from bpexts.utils import torch_allclose

# convolution parameters
in_channels = 3
out_channels = 2
kernel_size = (2, 2)
stride = (1, 1)
padding = (0, 0)
dilation = (1, 1)
bias = False

# predefined kernel matrix
kernel11 = [[1, 1], [2, 2]]
kernel12 = [[1, 1], [1, 1]]
kernel13 = [[0, 1], [1, 0]]
kernel21 = [[1, 0], [0, 1]]
kernel22 = [[2, 1], [2, 1]]
kernel23 = [[1, 2], [2, 0]]
kernel = Tensor([[kernel11, kernel12, kernel13],
                 [kernel21, kernel22, kernel23]]).float()

# input (1 sample)
in_feature1 = [[1, 2, 0], [1, 1, 3], [0, 2, 2]]
in_feature2 = [[0, 2, 1], [0, 3, 2], [1, 1, 0]]
in_feature3 = [[1, 2, 1], [0, 1, 3], [3, 3, 2]]
in1 = Tensor([[in_feature1, in_feature2, in_feature3]]).float()
result1 = [[14, 20], [15, 24]]
result2 = [[12, 24], [17, 26]]
out1 = Tensor([[result1, result2]]).float()

# convolution layer from torch.nn
conv2d = Conv2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    bias=bias)
conv2d.weight.data = kernel

# extended convolution layer
g_conv2d = G_Conv2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    bias=bias)
g_conv2d.weight.data = kernel

# as lists for zipping
inputs = [in1]
results = [out1]


def loss_function(tensor):
    """Test loss function. Sum over squared entries."""
    return ((tensor.contiguous().view(-1))**2).sum()


def test_forward():
    """Compare forward of torch.nn.Conv2d and exts.gradient.G_Conv2d.

    Handles only single instance batch.
    """
    for input, result in zip(inputs, results):
        out_conv2d = conv2d(input)
        assert torch_allclose(out_conv2d, result)
        out_g_conv2d = g_conv2d(input)
        assert torch_allclose(out_g_conv2d, result)


def random_convolutions_and_inputs(in_channels=None,
                                   out_channels=None,
                                   kernel_size=None,
                                   stride=None,
                                   padding=None,
                                   dilation=None,
                                   bias=None,
                                   kernel_shape=None,
                                   batch_size=None,
                                   in_size=None):
    """Return same torch/exts 2d conv modules and random inputs.

    Arguments can be fixed by handing them over.
    """
    # random convolution parameters
    in_channels = in_channels if in_channels is not None else randint(1, 3)
    out_channels = out_channels if out_channels is not None else randint(1, 3)
    kernel_size = kernel_size if kernel_size is not None\
        else (randint(1, 3), randint(1, 3))
    stride = stride if stride is not None else (randint(1, 3), randint(1, 3))
    padding = padding if padding is not None\
        else (randint(0, 2), randint(0, 2))
    dilation = dilation if dilation is not None\
        else (randint(1, 3), randint(1, 3))
    bias = bias if bias is not None else choice([True, False])
    # random kernel
    kernel_shape = (out_channels, in_channels) + kernel_size
    kernel = randn(kernel_shape)
    # random input
    batch_size = batch_size if batch_size is not None else randint(1, 3)
    in_size = in_size if in_size is not None\
        else (randint(8, 12), randint(8, 12))
    in_shape = (batch_size, in_channels) + in_size
    input = randn(in_shape)
    # torch.nn convolution
    conv2d = Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias)
    conv2d.weight.data = kernel
    # extended convolution layer
    g_conv2d = G_Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias)
    g_conv2d.weight.data = kernel
    # random bias
    if bias is True:
        bias_vals = randn(out_channels)
        conv2d.bias.data = bias_vals
        g_conv2d.bias.data = bias_vals
        assert torch_allclose(conv2d.bias, g_conv2d.bias)
    assert torch_allclose(conv2d.weight, g_conv2d.weight)
    return conv2d, g_conv2d, input


def compare_grads(conv2d, g_conv2d, input):
    """Feed input through nn and exts conv2d, compare bias/weight grad."""
    # backward for torch conv2d
    out_conv2d = conv2d(input)
    loss = loss_function(out_conv2d)
    loss.backward()
    # backward for exts conv2d
    out_g_conv2d = g_conv2d(input)
    loss_g = loss_function(out_g_conv2d)
    with config.bpexts(config.BATCH_GRAD):
        loss_g.backward()
    # need to choose lower precision for some reason
    assert torch_allclose(g_conv2d.bias.grad, conv2d.bias.grad, atol=1E-4)
    assert torch_allclose(g_conv2d.weight.grad, conv2d.weight.grad, atol=1E-4)
    assert torch_allclose(
        g_conv2d.bias.grad_batch.sum(0), conv2d.bias.grad, atol=1E-4)
    assert torch_allclose(
        g_conv2d.weight.grad_batch.sum(0), conv2d.weight.grad, atol=1E-4)


def test_random_grad(random_runs=10):
    """Compare bias gradients for a single sample."""
    for i in range(random_runs):
        conv2d, g_conv2d, input = random_convolutions_and_inputs(
            bias=True, batch_size=1)
        compare_grads(conv2d, g_conv2d, input)


def test_random_grad_batch(random_runs=10):
    """Check bias gradients for a batch."""
    for i in range(random_runs):
        conv2d, g_conv2d, input = random_convolutions_and_inputs(bias=True)
        compare_grads(conv2d, g_conv2d, input)
