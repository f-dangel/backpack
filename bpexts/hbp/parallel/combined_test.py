"""Test conversion/splitting HBPCompositionActivation to parallel."""

import torch
from torch import randn, Tensor, eye, zeros_like
from warnings import warn
from ..combined_relu import HBPReLULinear
from ..combined_sigmoid import HBPSigmoidLinear
from .combined import HBPParallelCompositionActivationLinear
from ...utils import (torch_allclose,
                      set_seeds,
                      memory_report)
from ...hessian import exact


test_classes = [HBPSigmoidLinear, HBPReLULinear]

in_features = 20
max_blocks = 3
in_features_list = [7, 7, 6]
out_features_list = [3, 3, 2]
out_features = 8
input = randn(1, in_features)


def random_input():
    """Return random input copy."""
    new_input = input.clone()
    new_input.requires_grad = True
    return new_input


def example_layer(cls):
    """Return example layer of HBPCompositionActivation."""
    set_seeds(0)
    return cls(in_features=in_features,
               out_features=out_features,
               bias=True)


def example_layer_parallel(cls, max_blocks=max_blocks):
    """Return example layer of HBPParallelCompositionActivation."""
    return HBPParallelCompositionActivationLinear(example_layer(cls),
                                                  max_blocks)


def test_num_blocks():
    """Test number of blocks."""
    for cls in test_classes:
        # smaller than out_features
        parallel = example_layer_parallel(cls, max_blocks=3)
        assert parallel.num_blocks == 3
        # smaller than out_features
        parallel = example_layer_parallel(cls, max_blocks=6)
        assert parallel.num_blocks == 4
        # larger than out_features
        parallel = example_layer_parallel(cls, max_blocks=10)
        assert parallel.num_blocks == out_features


def test_forward_pass():
    """Test whether parallel module is consistent with main module."""
    for cls in test_classes:
        layer = example_layer(cls)
        parallel = example_layer_parallel(cls, max_blocks=max_blocks)
        x = random_input()
        assert torch_allclose(layer(x), parallel(x))
        assert isinstance(parallel.main.linear.mean_input, Tensor)
        assert isinstance(parallel.main.activation.grad_phi, Tensor)
        assert isinstance(parallel.main.activation.gradgrad_phi, Tensor)
        assert not hasattr(parallel.main.activation, 'grad_output')
        # check HBP buffers
        for child in parallel.parallel_children():
            assert isinstance(child.linear.mean_input, Tensor)
            assert child.linear.mean_input is parallel.main.linear.mean_input
            assert not hasattr(child.activation, 'grad_phi')
            assert not hasattr(child.activation, 'gradgrad_phi')
            assert not hasattr(child.activation, 'grad_output')
 

def example_loss(tensor):
    """Sum all square entries of a tensor."""
    return (tensor**2).view(-1).sum(0)
 

def forward(layer, input):
    """Feed input through layers and loss. Return output and loss."""
    out = layer(input)
    return out, example_loss(out)


def test_backward_pass():
    """Test whether parallel module is consistent with main module."""
    for cls in test_classes:
        parallel = example_layer_parallel(cls, max_blocks=max_blocks)
        x = random_input()
        out, loss = forward(parallel, x)
        loss.backward()

        assert isinstance(parallel.main.activation.grad_output, Tensor)
        assert tuple(parallel.main.activation.grad_output.size()) == \
                    (1, in_features)
        # check HBP buffers
        for idx, child in enumerate(parallel.parallel_children()):
            assert not hasattr(child.activation, 'grad_output')
 

def input_hessian(cls):
    """Feed input through net and loss, backward the Hessian.

    Return the layers and input Hessian.
    """
    parallel = example_layer_parallel(cls)
    out, loss = forward(parallel, random_input())
    loss_hessian = 2 * eye(out.numel())
    loss.backward()
    # call HBP recursively
    out_h = loss_hessian
    in_h = parallel.backward_hessian(out_h,
                                     compute_input_hessian=True)
    return parallel, in_h


def brute_force_input_hessian(cls):
    """Compute Hessian of loss w.r.t. input."""
    parallel = example_layer_parallel(cls)
    input = random_input()
    _, loss = forward(parallel, input)
    return exact.exact_hessian(loss, [input])


def test_input_hessian():
    """Check if input Hessian is computed correctly."""
    for cls in test_classes:
        layer, in_h = input_hessian(cls)
        assert torch_allclose(in_h, brute_force_input_hessian(cls))
        for buffer in [layer.main.linear.weight,
                       layer.main.linear.bias]:
            assert buffer.grad is None
            assert not hasattr(buffer, 'hvp')
            assert not hasattr(buffer, 'hessian')
        for child in layer.parallel_children():
            for buffer in [child.linear.weight, child.linear.bias]:
                assert buffer.grad is not None
                assert hasattr(buffer, 'hvp')
                assert hasattr(buffer, 'hessian')
     

def brute_force_parameter_hessian(cls, which):
    """Compute Hessian of loss w.r.t. the weights."""
    layer = example_layer_parallel(cls)
    input = random_input()
    _, loss = forward(layer, input)
    if which == 'weight':
        return [exact.exact_hessian(loss, [child.linear.weight])
                for child in layer.parallel_children()]
    elif which == 'bias':
        return [exact.exact_hessian(loss, [child.linear.bias])
                for child in layer.parallel_children()]


def test_weight_hessian():
    """Check if weight Hessians are computed correctly."""
    for cls in test_classes:
        layer, _ = input_hessian(cls)
        for i, w_hessian in enumerate(
                brute_force_parameter_hessian(cls, 'weight')):
            w_h = layer._get_parallel_module(i).linear.weight.hessian()
            assert torch_allclose(w_h, w_hessian)


def test_bias_hessian():
    """Check if bias  Hessians are computed correctly."""
    for cls in test_classes:
        layer, _ = input_hessian(cls)
        for i, b_hessian in enumerate(
                brute_force_parameter_hessian(cls, 'bias')):
            b_h = layer._get_parallel_module(i).linear.bias.hessian
            assert torch_allclose(b_h, b_hessian)


def test_memory_consumption_vs_hbpcomposition():
    """Test if requires same amount of memory as HBPCompositionActivation."""
    for cls in test_classes:
        # HBPCompositionActivation
        layer = example_layer(cls)
        mem_stat1 = memory_report()
        # HBPParallelCompositionActivationLinear
        layer = example_layer_parallel(cls)
        mem_stat2 = memory_report()
        assert mem_stat1 == mem_stat2


def test_memory_consumption_forward():
    """Check memory consumption during splitting."""
    for cls in test_classes:
        # feed through HBPCompositionActivation
        layer = example_layer(cls)
        out = layer(random_input())
        mem_stat1 = memory_report()
        del layer, out
        # feed through HBPParallelCompositionActivationLinear
        layer = example_layer_parallel(cls)
        out = layer(random_input())
        mem_stat2 = memory_report()
        assert mem_stat1 == mem_stat2


def test_memory_consumption_during_hbp():
    """Check for constant memory consumption during HBP."""
    for cls in test_classes:
        # HBPParallelCompositionActivation layer
        parallel = example_layer_parallel(cls)
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


def compare_no_splitting_with_composition(device, cls, num_iters):
    """Check if parallel composition layer with splitting of 1 behaves
    exactly the same as HBPCompositionActivationLinear over multiple
    iterations of HBP.
    """
    linear = example_layer(cls).to(device)
    parallel = example_layer_parallel(cls, max_blocks=1).to(device)

    # check equality of parameters
    assert(len(list(linear.parameters())) ==
           len(list(parallel.parameters())) ==
           2)
    for p1, p2 in zip(linear.parameters(), parallel.parameters()):
        assert torch_allclose(p1, p2)

    # check equality of gradients/hvp over multiple runs
    for i in range(num_iters):
        # 5 samples
        input = randn(5, in_features, device=device)
        input.requires_grad = True
        # forward pass checking
        out1, loss1 = forward(linear, input)
        out2, loss2 = forward(parallel, input)
        assert torch_allclose(out1, out2)
        # gradient checking
        loss1.backward()
        loss2.backward()
        for p1, p2 in zip(linear.parameters(), parallel.parameters()):
            assert p1.grad is not None and p2.grad is not None
            assert not torch.allclose(p1.grad, zeros_like(p1))
            assert not torch.allclose(p2.grad, zeros_like(p2))
            assert torch_allclose(p1.grad, p2.grad)
        loss_hessian = randn(out_features, out_features, device=device)
        # check input Hessians and Hessian-vector products
        for mod2nd in ['zero', 'abs', 'clip', 'none']:
            # input Hessian
            in_h1 = linear.backward_hessian(loss_hessian,
                                            compute_input_hessian=True,
                                            modify_2nd_order_terms=mod2nd)
            in_h2 = parallel.backward_hessian(loss_hessian,
                                              compute_input_hessian=True,
                                              modify_2nd_order_terms=mod2nd)
            assert torch_allclose(in_h1, in_h2)
            # parameter Hessians
            for p1, p2 in zip(linear.parameters(), parallel.parameters()):
                v = randn(p1.numel(), device=device)
                assert torch_allclose(p1.hvp(v), p2.hvp(v))


def test_comparison_no_splitting_with_composition_cpu(num_iters=10):
    """Check if parallel single layer equals HBPLinear on CPU."""
    device = torch.device('cpu')
    for cls in test_classes:
        compare_no_splitting_with_composition(device, cls, num_iters)


def test_comparison_no_splitting_with_composition_gpu(num_iters=10):
    """Check if parallel single layer equals HBPLinear on GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        for cls in test_classes:
            compare_no_splitting_with_composition(device, cls, num_iters)
    else:
        warn('Could not find CUDA device')
