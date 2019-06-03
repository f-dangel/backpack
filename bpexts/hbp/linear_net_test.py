"""Test Hessian backpropagation of Linear and Sigmoid/ReLU.

The test network consists of three linear layers, each of them
followed by a nonlinear activation layer.
"""

from torch import (randn, eye)
from .linear import HBPLinear
from .sigmoid import HBPSigmoid
from .relu import HBPReLU
from ..hessian import exact
from ..utils import (torch_allclose, set_seeds)

in_features = [20, 10, 5]
out_features = [10, 5, 2]
input_size = (20, )
input = randn((1, ) + input_size)


def random_input():
    """Return random input copy."""
    return input.clone()


def create_layers(activation):
    """Create layers of the fully-connected network."""
    # same seed
    set_seeds(0)
    layers = []
    for (in_, out) in zip(in_features, out_features):
        layers.append(HBPLinear(in_features=in_, out_features=out))
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
        # ignore activation layers
        if idx % 2 == 1:
            continue
        b_brute_force = brute_force_hessian(-idx, 'bias', activation)
        # check bias Hessian-veector product
        for _ in range(random_vp):
            v = randn(layer.bias.numel())
            vp = layer.bias.hvp(v)
            vp_result = b_brute_force.matmul(v)
            assert torch_allclose(vp, vp_result, atol=1E-5)
    # test weight Hessians
    for idx, layer in enumerate(reversed(layers), 1):
        # ignore activation layers
        if idx % 2 == 1:
            continue
        w_brute_force = brute_force_hessian(-idx, 'weight', activation)
        # check weight Hessian-vector product
        for _ in range(random_vp):
            v = randn(layer.weight.numel())
            vp = layer.weight.hvp(v)
            vp_result = w_brute_force.matmul(v)
            assert torch_allclose(vp, vp_result, atol=1E-5)


def test_sigmoid_network_parameter_hessians(random_vp=10):
    """Check HBP Hessians and HVP for Sigmoid activations."""
    check_network_parameter_hessians(random_vp, HBPSigmoid)


def test_relu_network_parameter_hessians(random_vp=10):
    """Check HBP Hessians and HVP for ReLU activations."""
    check_network_parameter_hessians(random_vp, HBPReLU)
