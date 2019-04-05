"""Recursive Hessian-vector products in spirit of HBP for Max pooling in 2d."""

import collections
from torch import (randn, randint, eye)
from torch.nn import MaxPool2d
from .maxpool2d_recursive import HBPMaxPool2dRecursive
from ..hessian import exact
from ..utils import (torch_allclose, set_seeds)

in_channels = 3
kernel_size = (2, 2)
stride = None
padding = 0
dilation = 1

input_size = (8, 8)
input = randn((1, in_channels) + input_size)


def random_input():
    """Return random input copy."""
    return input.clone()


def create_layers():
    """Create the max pool layer."""
    return [
        HBPMaxPool2dRecursive(
            kernel_size, stride=stride, padding=padding, dilation=dilation)
    ]


def forward(layers, input):
    """Feed input through layers and loss. Return output and loss."""
    x = input
    for layer in layers:
        x = layer(x)

        y = layer.unpool2d(x)
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
    x = random_input()
    x.requires_grad = True
    out, loss = forward(layers, x)
    loss_hessian = 2 * eye(out.numel())
    loss.backward()
    # call HBP recursively
    out_h = loss_hessian
    for layer in reversed(layers):
        out_h = layer.backward_hessian(out_h)
    return layers, out_h


def test_input_hessian_vp(random_vp=10):
    """Test equality between input Hessian VP and brute force method."""
    _, out_hessian_vp = hessian_backward()

    layers = create_layers()
    x = random_input()
    x.requires_grad = True
    _, loss = forward(layers, x)
    x_brute_force = exact.exact_hessian(loss, [x])

    for _ in range(random_vp):
        v = randn(x.numel() // x.size()[0])
        vp = out_hessian_vp(v)
        vp_result = x_brute_force.matmul(v)
        assert torch_allclose(vp, vp_result, atol=1E-5)


def test_max_pool_indices():
    """Experiments with the index function of maxpool2d."""
    set_seeds(1)
    x = randint(0, 10, (2, 2, 4, 4))
    assert len(x.size()) == 4
    kernel_size = (2, 2)
    print("in\n", x)
    pool = MaxPool2d(
        kernel_size=kernel_size, stride=kernel_size, return_indices=True)

    pool2 = MaxPool2d(
        kernel_size=kernel_size, stride=kernel_size, return_indices=False)

    out, indices = pool(x)
    print(indices.size())
    out2 = pool2(x)
    print("out\n", out)
    print("indices\n", indices)
    print("out2\n", out)

    # indices in flattened input
    channel_elements = x.numel() // x.size(0) // x.size(1)
    for i in range(x.size()[1]):
        indices[:, i, :, :] += i * channel_elements

    print("indices in flattened quantity\n", indices)
