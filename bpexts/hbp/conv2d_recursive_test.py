"""Test recursive Hessian-vector products for conv2d layer."""

from .conv2d_recursive import HBPConv2dRecursive
from torch import (randn, eye)
from ..hessian import exact
from ..utils import (torch_allclose, set_seeds)

in_channels = [3]  #, 5, 4]
out_channels = [5]  # , 4, 2]
kernel_sizes = [(4, 4)]  # , (3, 2), (1, 3)]
paddings = [2]  #, 3, 1]
strides = [2]  #, 2, 1]

input_size = (7, 7)
input = randn((1, in_channels[0]) + input_size)


def random_input():
    """Return random input copy."""
    return input.clone()


def create_layers():
    """Create the convolution layers of the network."""
    # same seed
    set_seeds(0)
    layers = []
    for (in_, out, kernel, pad, stride) in zip(
            in_channels, out_channels, kernel_sizes, paddings, strides):
        layers.append(
            HBPConv2dRecursive(
                in_channels=in_,
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


def hessian_backward():
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


def test_bias_hessian_vp(random_vp=10):
    """Test equality between bias Hessian VP and brute force method."""
    # test bias Hessians
    layers = hessian_backward()
    for idx, layer in enumerate(reversed(layers), 1):
        assert not hasattr(layer.bias, "hessian")
        assert hasattr(layer.bias, "hvp")
        b_brute_force = brute_force_hessian(-idx, 'bias')
        # check bias Hessian-vector product
        for _ in range(random_vp):
            v = randn(layer.bias.numel())
            vp = layer.bias.hvp(v)
            vp_result = b_brute_force.matmul(v)
            assert torch_allclose(vp, vp_result, atol=1E-5)
    # TODO
    """
    # test weight Hessians
    for idx, layer in enumerate(reversed(layers), 1):
        w_hessian = layer.weight.hessian()
        w_brute_force = brute_force_hessian(-idx, 'weight')
        assert torch_allclose(w_hessian, w_brute_force, atol=1E-5)
        # check weight Hessian-vector product
        for _ in range(random_vp):
            v = randn(layer.weight.numel())
            vp = layer.weight.hvp(v)
            vp_result = w_brute_force.matmul(v)
            assert torch_allclose(vp, vp_result, atol=1E-5)
    """
