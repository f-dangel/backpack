"""Test Hessian backpropagation of sequence of identical parallel modules"""

from torch import randn, eye
from .parallel_sequential import HBPParallelIdenticalSequential
from ..sequential import HBPSequential
from ..combined_sigmoid import HBPSigmoidLinear
from ...utils import (torch_allclose,
                      set_seeds)
from ...hessian import exact

in_features = [20, 10]
out_features = [10, 8]
num_blocks = 3
num_layers = len(in_features)
input = randn(1, in_features[0])


def random_input():
    """Return random input copy."""
    input_with_grad = input.clone()
    input_with_grad.requires_grad = True
    return input_with_grad


def create_sequence():
    """Return sequence of identical modules in parallel."""
    # same seed
    set_seeds(0)
    layers = [HBPSigmoidLinear(in_, out, bias=True)
              for in_, out in zip(in_features, out_features)]
    sequence = HBPSequential(*layers)
    return HBPParallelIdenticalSequential.from_sequential(
            sequence).split_into_blocks(num_blocks)


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
    layer = create_sequence()
    x, loss = forward(layer, input)
    loss_hessian = 2 * eye(x.numel())
    loss.backward()
    # call HBP recursively
    out_h = loss_hessian
    layer.backward_hessian(out_h)
    return layer


def brute_force_hessian(layer_idx, parallel_idx, which):
    """Compute Hessian of loss w.r.t. parameter in layer."""
    sequence = create_sequence()
    _, loss = forward(sequence, random_input())
    if which == 'weight':
        return exact.exact_hessian(loss,
                                   [list(sequence.children())[layer_idx]
                                    .get_submodule(parallel_idx)
                                    .linear.weight])
    elif which == 'bias':
        return exact.exact_hessian(loss,
                                   [list(sequence.children())[layer_idx]
                                    .get_submodule(parallel_idx)
                                    .linear.bias])
    else:
        raise ValueError


def brute_force_input_hessian():
    """Compute the Hessian with respect to the input by brute force."""
    layer = create_sequence()
    input = random_input()
    input.requires_grad = True
    _, loss = forward(layer, input)
    return exact.exact_hessian(loss, [input])


def test_input_hessians():
    """Test whether Hessian with respect to input is correctly reproduced."""
    # need to unite for exact input Hessian
    layer = create_sequence().unite()
    out, loss = forward(layer, random_input())
    loss_hessian = 2 * eye(out.numel())
    loss.backward()
    # call HBP recursively
    in_h = layer.backward_hessian(loss_hessian, compute_input_hessian=True)
    assert torch_allclose(in_h, brute_force_input_hessian())


def test_parameter_hessians(random_vp=10):
    """Test equality between HBP Hessians and brute force Hessians.
    Check Hessian-vector products."""
    # test bias Hessians
    layer = hessian_backward()
    for idx in range(num_layers):
        layer_idx = list(layer.children())[idx]
        for n in range(num_blocks):
            linear = layer_idx.get_submodule(n).linear
            b_hessian = linear.bias.hessian
            b_brute_force = brute_force_hessian(idx, n, 'bias')
            assert torch_allclose(b_hessian, b_brute_force, atol=1E-5)
            # check bias Hessian-veector product
            for _ in range(random_vp):
                v = randn(linear.bias.numel())
            vp = linear.bias.hvp(v)
            vp_result = b_brute_force.matmul(v)
            assert torch_allclose(vp, vp_result, atol=1E-5)
    # test weight Hessians
    for idx in range(num_layers):
        layer_idx = list(layer.children())[idx]
        for n in range(num_blocks):
            linear = layer_idx.get_submodule(n).linear
            w_hessian = linear.weight.hessian()
            w_brute_force = brute_force_hessian(idx, n, 'weight')
            assert torch_allclose(w_hessian, w_brute_force, atol=1E-5)
            # check weight Hessian-vector product
            for _ in range(random_vp):
                v = randn(linear.weight.numel())
                vp = linear.weight.hvp(v)
                vp_result = w_brute_force.matmul(v)
                assert torch_allclose(vp, vp_result, atol=1E-5)
