"""Test conversion HBPLinear to parallel series and splitting."""

import torch
from torch import randn, Tensor, eye, zeros_like, ones_like
from torch.optim import SGD
from warnings import warn
from bpexts.hbp.linear import HBPLinear
from bpexts.hbp.parallel.linear import HBPParallelLinear
from bpexts.utils import (torch_contains_nan, set_seeds, memory_report)
from bpexts.optim.cg_newton import CGNewton
from bpexts.hessian import exact

in_features = 20
max_blocks = 3
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
    return HBPLinear(
        in_features=in_features, out_features=out_features, bias=True)


def example_linear_parallel(max_blocks=max_blocks):
    """Return example layer of HBPParallelLinear."""
    layer = HBPParallelLinear(example_linear(), max_blocks)
    print(layer)
    layer.disable_hbp()
    print(layer)
    layer.enable_hbp()
    print(layer)
    return layer


def test_num_blocks():
    """Test number of blocks."""
    # smaller than out_features
    parallel = example_linear_parallel(max_blocks=3)
    assert parallel.num_blocks == 3
    # smaller than out_features
    parallel = example_linear_parallel(max_blocks=5)
    assert parallel.num_blocks == 4
    # larger than out_features
    parallel = example_linear_parallel(max_blocks=10)
    assert parallel.num_blocks == out_features


def test_forward_pass():
    """Test whether parallel module is consistent with main module."""
    linear = example_linear()
    parallel = example_linear_parallel(max_blocks=max_blocks)
    x = random_input()
    assert torch.allclose(linear(x), parallel(x))
    hessian = torch.rand(2 * (linear.out_features, ))
    linear.backward_hessian(hessian)
    parallel.backward_hessian(hessian)
    assert isinstance(parallel.main.mean_input, Tensor)
    # check HBP buffers
    for child in parallel.parallel_children():
        assert hasattr(child, 'mean_input')
        assert child.mean_input is parallel.main.mean_input


def example_loss(tensor):
    """Sum all square entries of a tensor."""
    return ((tensor - 1)**2).view(-1).sum(0)


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
    in_h = parallel.backward_hessian(out_h, compute_input_hessian=True)
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
    assert torch.allclose(in_h, brute_force_input_hessian())
    for buffer in [layer.main.weight, layer.main.bias]:
        assert buffer.grad is None
        assert not hasattr(buffer, 'hvp')
    #layer.reference_gradients(layer, None, None)
    for child in layer.parallel_children():
        for buffer in [child.weight, child.bias]:
            assert buffer.grad is not None
            assert hasattr(buffer, 'hvp')


def brute_force_parameter_hessian(which):
    """Compute Hessian of loss w.r.t. the weights."""
    layer = example_linear_parallel()
    input = random_input()
    _, loss = forward(layer, input)
    if which == 'weight':
        return [
            exact.exact_hessian(loss, [child.weight])
            for child in layer.parallel_children()
        ]
    elif which == 'bias':
        return [
            exact.exact_hessian(loss, [child.bias])
            for child in layer.parallel_children()
        ]


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


def compare_no_splitting_with_linear(device, num_iters):
    """Check if parallel linear layer with splitting of 1 behaves
    exactly the same as HBPLinear over multiple iterations of HBP.
    """
    linear = example_linear().to(device)
    parallel = example_linear_parallel(max_blocks=1).to(device)

    # check equality of parameters
    assert (len(list(linear.parameters())) == len(list(parallel.parameters()))
            == 2)
    for p1, p2 in zip(linear.parameters(), parallel.parameters()):
        assert torch.allclose(p1, p2)

    # check equality of gradients/hvp over multiple runs
    for i in range(num_iters):
        # check input Hessians and Hessian-vector products
        for mod2nd in ['zero', 'abs', 'clip', 'none']:
            # 5 samples
            input = randn(5, in_features, device=device)
            # forward pass checking
            out1, loss1 = forward(linear, input)
            out2, loss2 = forward(parallel, input)
            assert torch.allclose(out1, out2)
            # gradient checking
            loss1.backward()
            loss2.backward()
            for p1, p2 in zip(linear.parameters(), parallel.parameters()):
                assert p1.grad is not None and p2.grad is not None
                assert not torch.allclose(p1.grad, zeros_like(p1))
                assert not torch.allclose(p2.grad, zeros_like(p2))
                assert torch.allclose(p1.grad, p2.grad)
            loss_hessian = randn(out_features, out_features, device=device)
            # input Hessian
            in_h1 = linear.backward_hessian(
                loss_hessian,
                compute_input_hessian=True,
                modify_2nd_order_terms=mod2nd)
            in_h2 = parallel.backward_hessian(
                loss_hessian,
                compute_input_hessian=True,
                modify_2nd_order_terms=mod2nd)
            assert torch.allclose(in_h1, in_h2)
            # parameter Hessians
            for p1, p2 in zip(linear.parameters(), parallel.parameters()):
                v = randn(p1.numel(), device=device)
                assert torch.allclose(p1.hvp(v), p2.hvp(v))


def test_comparison_no_splitting_with_linear_cpu(num_iters=10):
    """Check if parallel single layer equals HBPLinear on CPU."""
    device = torch.device('cpu')
    compare_no_splitting_with_linear(device, num_iters)


def test_comparison_no_splitting_with_linear_gpu(num_iters=10):
    """Check if parallel single layer equals HBPLinear on GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        compare_no_splitting_with_linear(device, num_iters)
    else:
        warn('Could not find CUDA device')


def optim_example_loss(tensor):
    """Loss function that does not lead to overflow
    during optimization."""
    return ((tensor - 0.9)**2).view(-1).mean()


def optim_forward(layer, input):
    """Compute forward pass through model and optim loss function."""
    output = layer(input)
    return output, optim_example_loss(output)


def compare_sgd_optimization_no_split_with_linear(device, num_iters):
    """Check if parallel linear layer with splitting of 1 behaves
    exactly the same as a normal HBPParallel over multiple
    iterations of SGD optimization.
    """
    # create layers, load to device
    linear = example_linear().to(device)
    parallel = example_linear_parallel(max_blocks=1).to(device)

    # check equality of parameters
    assert (len(list(linear.parameters())) == len(list(parallel.parameters()))
            == 2)

    # create optimizers
    opt1 = SGD(linear.parameters(), 0.1)
    opt2 = SGD(parallel.parameters(), 0.1)

    # check same behavior over multiple runs
    for i in range(num_iters):
        # 5 samples, noise around a vector of ones
        input = 0.2 * randn(5, in_features, device=device)
        input = input + ones_like(input)

        # forward pass checking
        out1, loss1 = optim_forward(linear, input)
        out2, loss2 = optim_forward(parallel, input)
        assert torch.allclose(out1, out2)
        assert torch.allclose(loss1, loss2)

        # optimization step
        opt1.zero_grad()
        opt2.zero_grad()
        loss1.backward()
        loss2.backward()
        opt1.step()
        opt2.step()

        # check for same grads, values, and no nans
        for p1, p2 in zip(linear.parameters(), parallel.parameters()):
            # nan checks
            assert not torch_contains_nan(p1)
            assert not torch_contains_nan(p1.grad)
            assert not torch_contains_nan(p2)
            assert not torch_contains_nan(p2.grad)
            # grad/value checks
            assert torch.allclose(p1.grad, p2.grad)
            assert torch.allclose(p1, p2)

        # model parameter checks: no nans and same values
        assert not torch_contains_nan(linear.weight)
        assert not torch_contains_nan(linear.bias)
        assert not torch_contains_nan(parallel.main.weight)
        assert not torch_contains_nan(parallel.main.bias)
        assert torch.allclose(linear.weight, parallel.main.weight)
        assert torch.allclose(linear.bias, parallel.main.bias)


def test_compare_sgd_optimization_no_split_with_linear_cpu(num_iters=50):
    """Check if parallel linear layer without splitting behaves
    the same as HBPLinear during optimization with SGD on CPU."""
    device = torch.device('cpu')
    compare_sgd_optimization_no_split_with_linear(device, num_iters)


def test_compare_sgd_optimization_no_split_with_linear_gpu(num_iters=50):
    """Check if parallel linear layer without splitting behaves
    the same as HBPLinear during optimization with SGD on GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        compare_sgd_optimization_no_split_with_linear(device, num_iters)
    else:
        warn('Could not find CUDA device')


def compare_newton_optimization_no_split_with_linear(device, num_iters):
    """Check if parallel linear layer with splitting of 1 behaves
    exactly the same as a normal HBPParallel over multiple
    iterations of SGD optimization.
    """
    # create layers, load to device
    linear = example_linear().to(device)
    parallel = example_linear_parallel(max_blocks=1).to(device)

    # check equality of parameters
    assert (len(list(linear.parameters())) == len(list(parallel.parameters()))
            == 2)

    # create optimizers
    opt1 = CGNewton(linear.parameters(), 0.1, 0.05, cg_tol=1E-1, cg_maxiter=50)
    opt2 = CGNewton(
        parallel.parameters(), 0.1, 0.05, cg_tol=1E-1, cg_maxiter=50)

    # check equality of gradients/hvp over multiple runs
    for i in range(num_iters):
        # 5 samples, noise around a vector of ones
        input = 0.2 * randn(5, in_features, device=device)
        input = input + ones_like(input)
        # forward pass checking
        out1, loss1 = optim_forward(linear, input)
        out2, loss2 = optim_forward(parallel, input)
        assert torch.allclose(out1, out2)
        assert torch.allclose(loss1, loss2)

        # model parameter checks: same values
        assert torch.allclose(linear.weight, parallel.main.weight)
        assert torch.allclose(linear.bias, parallel.main.bias)

        # computing gradients
        opt1.zero_grad()
        opt2.zero_grad()
        loss1.backward()
        loss2.backward()
        # backwarding Hessian
        loss_hessian = 2 * eye(out_features, device=device) / out1.numel()
        mod2nd = 'abs'
        # check input Hessian
        in_h1 = linear.backward_hessian(
            loss_hessian,
            compute_input_hessian=True,
            modify_2nd_order_terms=mod2nd)
        in_h2 = parallel.backward_hessian(
            loss_hessian,
            compute_input_hessian=True,
            modify_2nd_order_terms=mod2nd)
        assert torch.allclose(in_h1, in_h2)
        # check Hessian-vector products
        for p1, p2 in zip(linear.parameters(), parallel.parameters()):
            v = randn(p1.numel(), device=device)
            assert torch.allclose(p1.hvp(v), p2.hvp(v))
        # optimization step
        opt1.step()
        opt2.step()

        # check parameters and gradients for same values, no nans
        for p1, p2 in zip(linear.parameters(), parallel.parameters()):
            # nan checks
            assert not torch_contains_nan(p1)
            assert not torch_contains_nan(p2)
            assert not torch_contains_nan(p1.grad)
            assert not torch_contains_nan(p2.grad)
            # value checks
            assert torch.allclose(p1.grad, p2.grad)
            assert torch.allclose(p1, p2)

        # model parameter checks: no nans and same values
        assert not torch_contains_nan(linear.weight)
        assert not torch_contains_nan(linear.bias)
        assert not torch_contains_nan(parallel.main.weight)
        assert not torch_contains_nan(parallel.main.bias)
        assert torch.allclose(linear.weight, parallel.main.weight)
        assert torch.allclose(linear.bias, parallel.main.bias)


def test_compare_newton_optimization_no_split_with_linear_cpu(num_iters=50):
    """Check if parallel linear layer without splitting behaves the same
    as HBPLinear during optimization with Newton-style optimizer on CPU."""
    device = torch.device('cpu')
    compare_newton_optimization_no_split_with_linear(device, num_iters)


def test_compare_newton_optimization_no_split_with_linear_gpu(num_iters=50):
    """Check if parallel linear layer without splitting behaves the same
    as HBPLinear during optimization with Newton-style optimizer on GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        compare_newton_optimization_no_split_with_linear(device, num_iters)
    else:
        warn('Could not find CUDA device')
