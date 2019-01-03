"""Test conversion HBPLinear to parallel series and splitting."""

from torch import randn, Tensor, eye
from ..linear import HBPLinear
from .linear import HBPParallelLinear
from ...utils import (torch_allclose,
                      set_seeds)
from ...hessian import exact


in_features = 20
out_features_list = [2, 3, 4]
out_features_list2 = [1, 8]
num_layers = len(out_features_list)
input = randn(1, in_features)


def random_input():
    """Return random input copy."""
    new_input = input.clone()
    new_input.requires_grad=True
    return new_input


def test_forward_pass_hbp_linear():
    """Test whether single/split modules are consistent in forward."""
    linear = HBPLinear(in_features=in_features,
                       out_features=sum(out_features_list),
                       bias=True)
    x = random_input()
    # after construction with a single child
    parallel = HBPParallelLinear(linear)
    assert torch_allclose(linear(x), parallel(x))

    # after splitting
    parallel2 = parallel.split(out_features_list)
    assert torch_allclose(linear(x), parallel2(x))

    # after unite
    parallel3 = parallel2.unite()
    assert torch_allclose(linear.bias,
                          parallel3.get_submodule(0).bias)
    assert torch_allclose(linear.weight,
                          parallel3.get_submodule(0).weight)
    assert torch_allclose(linear(x), parallel3(x))

    # after splitting into number of blocks
    parallel4 = parallel2.split_into_blocks(4)
    assert torch_allclose(linear(x), parallel4(x))


def test_buffer_ref_hbp_linear_unite():
    """Check if buffer is correctly referenced when uniting."""
    linear = HBPLinear(in_features=in_features,
                       out_features=sum(out_features_list),
                       bias=True)
    x = random_input()

    # after construction with a single child
    parallel = HBPParallelLinear(linear)
    parallel(x)
    # check if buffer is shared
    assert isinstance(parallel.mean_input, Tensor)
    for mod in parallel.children():
        assert isinstance(mod.mean_input, Tensor)
        assert mod.mean_input is parallel.mean_input

    # after splitting
    parallel2 = parallel.split(out_features_list)
    parallel2(x)
    # check if buffer is shared
    assert isinstance(parallel2.mean_input, Tensor)
    for mod in parallel2.children():
        assert isinstance(mod.mean_input, Tensor)
        assert mod.mean_input is parallel2.mean_input

    # after unite
    parallel3 = parallel2.unite()
    parallel3(x)
    # check if buffer is shared
    assert isinstance(parallel3.mean_input, Tensor)
    for mod in parallel3.children():
        assert isinstance(mod.mean_input, Tensor)
        assert mod.mean_input is parallel3.mean_input

    # after splitting into number of blocks
    parallel4 = parallel2.split_into_blocks(4)
    parallel4(x)
    # check if buffer is shared
    assert isinstance(parallel4.mean_input, Tensor)
    for mod in parallel4.children():
        assert isinstance(mod.mean_input, Tensor)
        assert mod.mean_input is parallel4.mean_input


def example_layer():
    """Return example split layer."""
    set_seeds(0)
    return HBPParallelLinear(*[HBPLinear(in_features=in_features,
                                         out_features=out,
                                         bias=True)
                               for out in out_features_list])


def example_loss(tensor):
    """Sum all square entries of a tensor."""
    return (tensor**2).view(-1).sum(0)


def forward(layer, input):
    """Feed input through layers and loss. Return output and loss."""
    out = layer(input)
    return out, example_loss(out)


def input_hessian():
    """Feed input through net and loss, backward the Hessian.

    Return the layers and input Hessian.
    """
    layer = example_layer()
    x, loss = forward(layer, random_input())
    loss_hessian = 2 * eye(x.numel())
    loss.backward()
    # call HBP recursively
    out_h = loss_hessian
    in_h = layer.backward_hessian(out_h,
                                  compute_input_hessian=True)
    return layer, in_h


def brute_force_input_hessian():
    """Compute Hessian of loss w.r.t. input."""
    layer = example_layer()
    input = random_input()
    _, loss = forward(layer, input)
    return exact.exact_hessian(loss, [input])


def test_input_hessian():
    """Check if input Hessian is computed correctly."""
    _, in_h = input_hessian()
    assert torch_allclose(in_h, brute_force_input_hessian())


def brute_force_parameter_hessian(which):
    """Compute Hessian of loss w.r.t. the weights."""
    layer = example_layer()
    input = random_input()
    _, loss = forward(layer, input)
    if which == 'weight':
        return [exact.exact_hessian(loss, [child.weight])
                for child in layer.children()]
    elif which == 'bias':
        return [exact.exact_hessian(loss, [child.bias])
                for child in layer.children()]


def test_weight_hessian():
    """Check if weight Hessians are computed correctly."""
    layer, _ = input_hessian()
    for i, w_hessian in enumerate(
            brute_force_parameter_hessian('weight')):
        w_h = layer.get_submodule(i).weight.hessian()
        assert torch_allclose(w_h, w_hessian)


def test_bias_hessian():
    """Check if bias  Hessians are computed correctly."""
    layer, _ = input_hessian()
    for i, b_hessian in enumerate(
            brute_force_parameter_hessian('bias')):
        b_h = layer.get_submodule(i).bias.hessian
        assert torch_allclose(b_h, b_hessian)
