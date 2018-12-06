"""Test Hessian backpropagation of Conv2d for an entire net.

The test network consists of three convolutional layers.
"""

from torch import (randn, eye)
from .conv2d import HBPConv2d
from ..hessian import exact
from ..utils import (torch_allclose,
                     set_seeds)


in_channels = [2, 2, 2]
out_channels = [2, 2, 2]
kernel_sizes = [(3, 3), (2, 2), (1, 2)]
# BUG: Cannot handle non-zero padding in intermediate layers,
# probably in input Hessian computation
# The bug can be reproduced by setting paddings to [2, 1, 1]
paddings = [2, 0, 0]
strides = [2, 2, 1]

input_size = (10, 10)
input = randn((1, in_channels[0]) + input_size)


def random_input():
    """Return random input copy."""
    return input.clone()


def create_layers():
    """Create the convolution layers of the network."""
    # same seed
    set_seeds(0)
    layers = []
    for (in_,
         out,
         kernel,
         pad,
         stride) in zip(in_channels,
                        out_channels,
                        kernel_sizes,
                        paddings,
                        strides):
        layers.append(HBPConv2d(in_channels=in_,
                                out_channels=out,
                                kernel_size=kernel,
                                stride=stride,
                                padding=pad))
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


def hessian_backward(layers, input):
    """Feed input through net and loss, backward the Hessian.

    Return the layers.
    """
    layers = create_layers()
    x, loss = forward(layers, random_input())
    loss_hessian = 2 * eye(x.numel())
    loss.backward()
    # call HBP recursively
    out_h = loss_hessian
    for layer in reversed(layers):
        out_h = layer.backward_hessian(out_h)
    return layers


def brute_force_hessian(layer_idx, which):
    """Compute Hessian of loss w.r.t. parameter in layer."""
    layers = create_layers()
    _, loss = forward(layers, random_input())
    if which == 'weight':
        return exact.exact_hessian(loss, [layers[layer_idx].weight])
    elif which == 'bias':
        return exact.exact_hessian(loss, [layers[layer_idx].bias])
    else:
        raise ValueError


def test_network_parameter_hessians():
    """Test equality between HBP Hessians and brute force Hessians."""
    # test bias Hessians
    layers = hessian_backward(create_layers(), random_input())
    for idx, layer in enumerate(reversed(layers), 1):
        print('bias')
        print(-idx)
        b_hessian = layer.bias.hessian
        print(b_hessian)
        result = brute_force_hessian(-idx, 'bias')
        print(result)
        assert torch_allclose(b_hessian, result, atol=1E-5)
        print('ok')
    # test weight Hessians
    for idx, layer in enumerate(reversed(layers), 1):
        print('weight')
        print(-idx)
        w_hessian = layer.weight.hessian()
        print(w_hessian)
        result = brute_force_hessian(-idx, 'weight')
        print(result)
        assert torch_allclose(w_hessian, result, atol=1E-5)
