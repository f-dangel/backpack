"""Test conversion HBPLinear to parallel series and splitting."""

from torch import randn, Tensor, eye
from ..linear import HBPLinear
from .linear import HBPParallelLinear
from ...utils import (torch_allclose,
                      set_seeds,
                      memory_report)
from ...hessian import exact


in_features = 20
num_blocks = 3
out_features_list = [3, 3, 2]
out_features = 8
input = randn(1, in_features)


def random_input():
    """Return random input copy."""
    new_input = input.clone()
    new_input.requires_grad = True
    return new_input


def example_linear():
    """Return example layer of HBPLinear."""
    set_seeds(0)
    return HBPLinear(in_features=in_features,
                     out_features=out_features,
                     bias=True)

def example_linear_parallel(num_blocks=num_blocks):
    """Return example layer of HBPParallelLinear."""
    return HBPParallelLinear(example_linear(), num_blocks)


def test_num_blocks():
    """Test number of blocks."""
    # smaller than out_features
    parallel = example_linear_parallel(num_blocks=3)
    assert parallel.num_blocks == 3
    # larger than out_features
    parallel = example_linear_parallel(num_blocks=10)
    assert parallel.num_blocks == out_features 


def test_forward_pass():
    """Test whether parallel module is consistent with main module."""
    linear = example_linear()
    parallel = example_linear_parallel(num_blocks=num_blocks)
    x = random_input()
    assert torch_allclose(linear(x), parallel(x))
    assert isinstance(parallel.main.mean_input, Tensor)
    # check HBP buffers
    for child in parallel.parallel_children():
        assert isinstance(child.mean_input, Tensor)
        assert child.mean_input is parallel.main.mean_input


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
    parallel = example_linear_parallel()
    out, loss = forward(parallel, random_input())
    loss_hessian = 2 * eye(out.numel())
    loss.backward()
    # call HBP recursively
    out_h = loss_hessian
    in_h = parallel.backward_hessian(out_h,
                                     compute_input_hessian=True)
    return parallel, in_h


def brute_force_input_hessian():
    """Compute Hessian of loss w.r.t. input."""
    parallel = example_linear_parallel()
    input = random_input()
    _, loss = forward(parallel, input)
    return exact.exact_hessian(loss, [input])


def test_input_hessian():
    """Check if input Hessian is computed correctly."""
    layer, in_h = input_hessian()
    assert torch_allclose(in_h, brute_force_input_hessian())
    for buffer in [layer.main.weight,
                   layer.main.bias]:
        assert buffer.grad is None
        assert not hasattr(buffer, 'hvp')
        assert not hasattr(buffer, 'hessian')
    for child in layer.parallel_children():
        for buffer in [child.weight, child.bias]:
            assert buffer.grad is not None
            assert hasattr(buffer, 'hvp')
            assert hasattr(buffer, 'hessian')
 

def brute_force_parameter_hessian(which):
    """Compute Hessian of loss w.r.t. the weights."""
    layer = example_linear_parallel()
    input = random_input()
    _, loss = forward(layer, input)
    if which == 'weight':
        return [exact.exact_hessian(loss, [child.weight])
                for child in layer.parallel_children()]
    elif which == 'bias':
        return [exact.exact_hessian(loss, [child.bias])
                for child in layer.parallel_children()]


def test_weight_hessian():
    """Check if weight Hessians are computed correctly."""
    layer, _ = input_hessian()
    for i, w_hessian in enumerate(
            brute_force_parameter_hessian('weight')):
        w_h = layer._get_parallel_module(i).weight.hessian()
        assert torch_allclose(w_h, w_hessian)


def test_bias_hessian():
    """Check if bias  Hessians are computed correctly."""
    layer, _ = input_hessian()
    for i, b_hessian in enumerate(
            brute_force_parameter_hessian('bias')):
        b_h = layer._get_parallel_module(i).bias.hessian
        assert torch_allclose(b_h, b_hessian)


def test_memory_consumption_vs_hbplinear():
    """Test if requires same amount of memory as HBPLinear."""
    # HBPLinear
    layer = example_linear()
    mem_stat1 = memory_report()
    # HBPParallelLinear
    layer = example_linear_parallel()
    mem_stat2 = memory_report()
    assert mem_stat1 == mem_stat2


def test_memory_consumption_forward():
    """Check memory consumption during splitting."""
    # feed through HBPLinear
    layer = example_linear()
    out = layer(random_input())
    mem_stat1 = memory_report()
    del layer, out
    # feed through HBPParallelLinear
    layer = example_linear_parallel()
    out = layer(random_input())
    mem_stat2 = memory_report()
    assert mem_stat1 == mem_stat2


def test_memory_consumption_during_hbp():
    """Check for constant memory consumption during HBP."""
    # HBPParallel layer
    parallel = example_linear_parallel()
    # will be initialized at first/10th HBP run
    mem_stat, mem_stat_after = None, None
    # HBP run
    for i in range(10):
        input = random_input()
        out, loss = forward(parallel, input)
        loss_hessian = 2 * eye(out.numel())
        loss.backward()
        out_h = loss_hessian
        parallel.backward_hessian(out_h)
        if i == 0:
            mem_stat = memory_report()
        if i == 9:
            mem_stat_after = memory_report()
    assert mem_stat == mem_stat_after
