"""Test Hessian backpropagation of Conv2d for an entire net.

The test network consists of three convolutional layers.
"""

from torch import (randn, eye)
from .conv2d import HBPConv2d
from .sigmoid import HBPSigmoid
from .relu import HBPReLU
from ..hessian import exact
from ..utils import (torch_allclose,
                     set_seeds)


in_channels = [3, 5, 4]
out_channels = [5, 4, 2]
kernel_sizes = [(4, 4), (3, 2), (1, 3)]
paddings = [2, 3, 1]
strides = [2, 2, 1]

input_size = (7, 7)
input = randn((1, in_channels[0]) + input_size)


def random_input():
    """Return random input copy."""
    return input.clone()


def create_layers(activation):
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


def hessian_backward(activation):
    """Feed input through net and loss, backward the Hessian.

    Return the layers.
    """
    layers = create_layers(activation)
    x, loss = forward(layers, random_input())
    loss_hessian = 2 * eye(x.numel())
    loss.backward()
    # call HBP recursively
    out_h = loss_hessian
    for layer in reversed(layers):
        out_h = layer.backward_hessian(out_h)
    return layers


def brute_force_hessian(layer_idx, which, activation):
    """Compute Hessian of loss w.r.t. parameter in layer."""
    layers = create_layers(activation)
    _, loss = forward(layers, random_input())
    if which == 'weight':
        return exact.exact_hessian(loss, [layers[layer_idx].weight])
    elif which == 'bias':
        return exact.exact_hessian(loss, [layers[layer_idx].bias])
    else:
        raise ValueError


def check_network_parameter_hessians(random_vp, activation):
    """Test equality between HBP Hessians and brute force Hessians.
    Check Hessian-vector products."""
    # test bias Hessians
    layers = hessian_backward(activation)
    for idx, layer in enumerate(reversed(layers), 1):
        # skip activation layers
        if idx % 2 == 1:
            continue
        b_hessian = layer.bias.hessian
        b_brute_force = brute_force_hessian(-idx, 'bias', activation)
        assert torch_allclose(b_hessian, b_brute_force, atol=1E-5)
        # check bias Hessian-veector product
        for _ in range(random_vp):
            v = randn(layer.bias.numel())
            vp = layer.bias.hvp(v)
            vp_result = b_brute_force.matmul(v)
            assert torch_allclose(vp, vp_result, atol=1E-5)
    # test weight Hessians
    for idx, layer in enumerate(reversed(layers), 1):
        # skip activation layers
        if idx % 2 == 1:
            continue
        w_hessian = layer.weight.hessian()
        w_brute_force = brute_force_hessian(-idx, 'weight', activation)
        assert torch_allclose(w_hessian, w_brute_force, atol=1E-5)
        # check weight Hessian-vector product
        for _ in range(random_vp):
            v = randn(layer.weight.numel())
            vp = layer.weight.hvp(v)
            vp_result = w_brute_force.matmul(v)
            assert torch_allclose(vp, vp_result, atol=1E-5)


def test_network_parameter_hessians_sigmoid():
    """Test HBP and HVP for parameters in Sigmoid-activated conv net."""
    check_network_parameter_hessians(10, HBPSigmoid)


def test_network_parameter_hessians_relu():
    """Test HBP and HVP for parameters in ReLU-activated conv net."""
    check_network_parameter_hessians(10, HBPReLU)
