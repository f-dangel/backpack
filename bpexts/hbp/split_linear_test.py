"""Test Hessian backpropagation of split linear layer."""

from torch import randn, eye
from .split_linear import HBPSplitLinear
from ..utils import (torch_allclose,
                     set_seeds)
from ..hessian import exact

in_features_list = [3, 4, 5]
out_features_list = [2, 4, 6]
num_layers = len(in_features_list)
input_size = sum(in_features_list)
input = randn(1, input_size)


def random_input():
    """Return random input copy."""
    return input.clone()


def create_layer():
    """Return example linear layer."""
    # same seed
    set_seeds(0)
    return HBPSplitLinear(in_features_list=in_features_list,
                          out_features_list=out_features_list,
                          bias=True)


def forward(layer, input):
    """Feed input through layer and loss. Return output and loss."""
    output = layer(input)
    return output, example_loss(output)


def example_loss(tensor):
    """Test loss function. Sum over squared entries.

    The Hessian of this function with respect to its
    inputs is given by an identity matrix scaled by 2.
    """
    return (tensor**2).contiguous().view(-1).sum()


def hessian_backward():
    """Feed input through layer and loss, backward the Hessian.

    Return the layer.
    """
    layer = create_layer()
    x, loss = forward(layer, random_input())
    loss_hessian = 2 * eye(x.numel())
    loss.backward()
    # call HBP recursively
    out_h = loss_hessian
    layer.backward_hessian(out_h)
    return layer


def brute_force_hessian(layer_idx, which):
    """Compute Hessian of loss w.r.t. parameter in layer."""
    layer = create_layer()
    _, loss = forward(layer, random_input())
    if which == 'weight':
        return exact.exact_hessian(loss,
                                   [layer.get_submodule(layer_idx).weight])
    elif which == 'bias':
        return exact.exact_hessian(loss,
                                   [layer.get_submodule(layer_idx).bias])
    else:
        raise ValueError


def test_parameter_hessians(random_vp=10):
    """Test equality between HBP Hessians and brute force Hessians.
    Check Hessian-vector products."""
    # test bias Hessians
    layer = hessian_backward()
    for idx in range(num_layers):
        b_hessian = layer.get_submodule(idx).bias.hessian
        b_brute_force = brute_force_hessian(idx, 'bias')
        assert torch_allclose(b_hessian, b_brute_force, atol=1E-5)
        # check bias Hessian-veector product
        for _ in range(random_vp):
            v = randn(layer.get_submodule(idx).bias.numel())
            vp = layer.get_submodule(idx).bias.hvp(v)
            vp_result = b_brute_force.matmul(v)
            assert torch_allclose(vp, vp_result, atol=1E-5)
    # test weight Hessians
    for idx in range(num_layers):
        w_hessian = layer.get_submodule(idx).weight.hessian()
        w_brute_force = brute_force_hessian(idx, 'weight')
        assert torch_allclose(w_hessian, w_brute_force, atol=1E-5)
        # check weight Hessian-vector product
        for _ in range(random_vp):
            v = randn(layer.get_submodule(idx).weight.numel())
            vp = layer.get_submodule(idx).weight.hvp(v)
            vp_result = w_brute_force.matmul(v)
            assert torch_allclose(vp, vp_result, atol=1E-5)
