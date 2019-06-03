"""Test Hessian backpropagation for a sequence of layers."""

from torch import randn, eye
from .sequential import HBPSequential
from .sigmoid import HBPSigmoid
from .linear import HBPLinear
from ..utils import (torch_allclose, set_seeds)
from ..hessian import exact

in_features_list = [10, 5, 4]
out_features_list = [5, 4, 2]

input = randn(1, in_features_list[0])


def random_input():
    """Return random input copy."""
    return input.clone()


def create_layers():
    """Return example layers for a sequential network."""
    # same seed
    set_seeds(0)
    layers = []
    for in_, out in zip(in_features_list, out_features_list):
        layers.append(HBPLinear(in_features=in_, out_features=out, bias=True))
        layers.append(HBPSigmoid())
    return HBPSequential(*layers)


def example_loss(tensor):
    """Test loss function. Sum over squared entries.

    The Hessian of this function with respect to its
    inputs is given by an identity matrix scaled by 2.
    """
    return (tensor**2).contiguous().view(-1).sum()


def brute_force_hessian(layer_idx, which):
    """Compute Hessian of loss w.r.t. parameter in layer."""
    sequence = create_layers()
    out = sequence(random_input())
    loss = example_loss(out)
    if which == 'weight':
        return exact.exact_hessian(loss, [sequence[layer_idx].weight])
    elif which == 'bias':
        return exact.exact_hessian(loss, [sequence[layer_idx].bias])
    else:
        raise ValueError


def hessian_backward():
    """Feed input through layer and loss, backward the Hessian.

    Return the layer.
    """
    sequence = create_layers()
    out = sequence(random_input())
    loss = example_loss(out)
    loss_hessian = 2 * eye(out.numel())
    loss.backward()
    in_h = sequence.backward_hessian(loss_hessian)
    assert in_h is None
    return sequence


def test_parameter_hessians(random_vp=10):
    """Test equality between HBP Hessians and brute force Hessians.
    Check Hessian-vector products."""
    # test bias Hessians
    sequence = hessian_backward()
    for idx in range(0, len(sequence), 2):
        b_brute_force = brute_force_hessian(idx, 'bias')
        # check bias Hessian-veector product
        for _ in range(random_vp):
            v = randn(sequence[idx].bias.numel())
            vp = sequence[idx].bias.hvp(v)
            vp_result = b_brute_force.matmul(v)
            assert torch_allclose(vp, vp_result, atol=1E-5)
    # test weight Hessians
    for idx in range(0, len(sequence), 2):
        w_brute_force = brute_force_hessian(idx, 'weight')
        # check weight Hessian-vector product
        for _ in range(random_vp):
            v = randn(sequence[idx].weight.numel())
            vp = sequence[idx].weight.hvp(v)
            vp_result = w_brute_force.matmul(v)
            assert torch_allclose(vp, vp_result, atol=1E-5)
