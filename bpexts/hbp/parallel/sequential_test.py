"""Test Hessian backpropagation of sequence of identical parallel modules"""

import torch
from torch import randn, eye, zeros_like
from warnings import warn
from .sequential import HBPParallelSequential
from ..sequential import HBPSequential
from ..linear import HBPLinear
from .linear import HBPParallelLinear
from ..combined_sigmoid import HBPSigmoidLinear
from ..combined_relu import HBPReLULinear
from ..sigmoid import HBPSigmoid
from ..relu import HBPReLU
from .combined import HBPParallelCompositionActivationLinear
from ...utils import (set_seeds, memory_report)
from ...optim.cg_newton import CGNewton
from ...hessian import exact

in_features = [20, 15, 10]
out_features = [15, 10, 5]
# DO NOT MODIFY!
classes = [HBPLinear, HBPSigmoidLinear, HBPReLULinear, HBPSigmoid, HBPReLU]
# DO NOT MODIFY!
target_classes = [
    HBPParallelLinear, HBPParallelCompositionActivationLinear,
    HBPParallelCompositionActivationLinear, HBPSigmoid, HBPReLU
]
max_blocks = 2
num_layers = len(in_features)
input = randn(1, in_features[0])


def random_input():
    """Return random input copy."""
    input_with_grad = input.clone()
    input_with_grad.requires_grad = True
    return input_with_grad


def create_layers():
    """Create list of layers for the example sequence."""
    set_seeds(0)
    layers = []
    for cls, in_, out in zip(classes, in_features, out_features):
        layers.append(cls(in_features=in_, out_features=out, bias=True))
    for cls in classes[3:]:
        layers.append(cls())
    return layers


def example_sequence():
    """Return example layer of HBPSequential."""
    return HBPSequential(*create_layers())


def example_sequence_parallel(max_blocks=max_blocks):
    """Return example layer of HBPParallelCompositionActivation."""
    return HBPParallelSequential(max_blocks, *create_layers())


def test_conversion():
    """Test conversion during initialization."""
    sequence = example_sequence_parallel()
    for mod, cls in zip(sequence.children(), target_classes):
        assert issubclass(mod.__class__, cls)


def test_num_blocks():
    """Test number of blocks."""
    # still possible for each layer
    sequence = example_sequence_parallel(3)
    for idx, parallel in enumerate(sequence.children()):
        if idx < 3:
            assert parallel.num_blocks == 3


def test_forward_pass():
    """Test for consistent behavior in forward pass."""
    x = random_input()
    sequence = example_sequence()
    parallel = example_sequence_parallel()
    assert torch.allclose(sequence(x), parallel(x))


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
    layer = example_sequence_parallel()
    x, loss = forward(layer, input)
    loss_hessian = 2 * eye(x.numel())
    loss.backward()
    # call HBP recursively
    out_h = loss_hessian
    layer.backward_hessian(out_h)
    return layer


def brute_force_hessian(layer_idx, parallel_idx, which):
    """Compute Hessian of loss w.r.t. parameter in layer."""
    if layer_idx > 2:
        raise ValueError('Works only for indices from 0 to 2')
    parallel = example_sequence_parallel()
    _, loss = forward(parallel, random_input())
    if which == 'weight':
        if layer_idx == 0:
            return exact.exact_hessian(loss, [
                list(parallel.children())[layer_idx]._get_parallel_module(
                    parallel_idx).weight
            ])
        else:
            return exact.exact_hessian(loss, [
                list(parallel.children())[layer_idx]._get_parallel_module(
                    parallel_idx).linear.weight
            ])
    elif which == 'bias':
        if layer_idx == 0:
            return exact.exact_hessian(loss, [
                list(parallel.children())[layer_idx]._get_parallel_module(
                    parallel_idx).bias
            ])
        else:
            return exact.exact_hessian(loss, [
                list(parallel.children())[layer_idx]._get_parallel_module(
                    parallel_idx).linear.bias
            ])
    else:
        raise ValueError


def brute_force_input_hessian():
    """Compute the Hessian with respect to the input by brute force."""
    layer = example_sequence_parallel()
    input = random_input()
    _, loss = forward(layer, input)
    return exact.exact_hessian(loss, [input])


def test_input_hessians():
    """Test whether Hessian with respect to input is correctly reproduced."""
    layer = example_sequence_parallel()
    out, loss = forward(layer, random_input())
    loss_hessian = 2 * eye(out.numel())
    loss.backward()
    # call HBP recursively
    in_h = layer.backward_hessian(loss_hessian, compute_input_hessian=True)
    assert torch.allclose(in_h, brute_force_input_hessian())


def test_parameter_hessians(random_vp=10):
    """Test equality between HBP Hessians and brute force Hessians.
    Check Hessian-vector products."""
    # test bias Hessians
    layer = hessian_backward()
    for idx, mod in enumerate(layer.children()):
        if idx > 2:
            continue
        layer_idx = list(layer.children())[idx]
        for n in range(layer_idx.num_blocks):
            linear = None
            if idx == 0:
                linear = layer_idx._get_parallel_module(n)
            else:
                linear = layer_idx._get_parallel_module(n).linear
            b_brute_force = brute_force_hessian(idx, n, 'bias')
            # check bias Hessian-veector product
            for _ in range(random_vp):
                v = randn(linear.bias.numel())
            vp = linear.bias.hvp(v)
            vp_result = b_brute_force.matmul(v)
            assert torch.allclose(vp, vp_result, atol=1E-5)
    # test weight Hessians
    for idx, mod in enumerate(layer.children()):
        if idx > 2:
            continue
        layer_idx = list(layer.children())[idx]
        for n in range(layer_idx.num_blocks):
            linear = None
            if idx == 0:
                linear = layer_idx._get_parallel_module(n)
            else:
                linear = layer_idx._get_parallel_module(n).linear
            w_brute_force = brute_force_hessian(idx, n, 'weight')
            # check weight Hessian-vector product
            for _ in range(random_vp):
                v = randn(linear.weight.numel())
                vp = linear.weight.hvp(v)
                vp_result = w_brute_force.matmul(v)
                assert torch.allclose(vp, vp_result, atol=1E-5)


def test_memory_consumption_before_backward_hessian():
    """Check memory consumption during splitting."""
    layer = example_sequence()
    print('No splitting')
    mem_stat1 = memory_report()

    layer = example_sequence_parallel()
    print('With splitting')
    mem_stat2 = memory_report()
    assert mem_stat1 == mem_stat2

    layer = example_sequence_parallel(10)
    print('With splitting')
    mem_stat2 = memory_report()
    assert mem_stat1 == mem_stat2


def test_memory_consumption_during_hbp():
    """Check memory consumption during Hessian backpropagation."""
    parallel = example_sequence_parallel()

    memory_report()

    mem_stat = None
    mem_stat_after = None

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


def compare_no_splitting_with_sequence(device, num_iters):
    """Check if parallel sequence with splitting of 1 behaves
    exactly the same as a normal sequence over multiple
    iterations of HBP.
    """
    sequence = example_sequence().to(device)
    parallel = example_sequence_parallel(max_blocks=1).to(device)

    # check equality of parameters
    assert (len(list(sequence.parameters())) == len(
        list(parallel.parameters())) == 6)
    for p1, p2 in zip(sequence.parameters(), parallel.parameters()):
        assert torch.allclose(p1, p2)

    # check equality of gradients/hvp over multiple runs
    for i in range(num_iters):
        # 5 samples
        input = randn(5, in_features[0], device=device)
        input.requires_grad = True
        # forward pass checking
        out1, loss1 = forward(sequence, input)
        out2, loss2 = forward(parallel, input)
        assert torch.allclose(out1, out2)
        # gradient checking
        loss1.backward()
        loss2.backward()
        for p1, p2 in zip(sequence.parameters(), parallel.parameters()):
            assert p1.grad is not None and p2.grad is not None
            assert not torch.allclose(p1.grad, zeros_like(p1))
            assert not torch.allclose(p2.grad, zeros_like(p2))
            assert torch.allclose(p1.grad, p2.grad)
        loss_hessian = randn(out_features[-1], out_features[-1], device=device)
        # check input Hessians and Hessian-vector products
        for mod2nd in ['zero', 'abs', 'clip', 'none']:
            # input Hessian
            in_h1 = sequence.backward_hessian(
                loss_hessian,
                compute_input_hessian=True,
                modify_2nd_order_terms=mod2nd)
            in_h2 = parallel.backward_hessian(
                loss_hessian,
                compute_input_hessian=True,
                modify_2nd_order_terms=mod2nd)
            assert torch.allclose(in_h1, in_h2)
            # parameter Hessians
            for p1, p2 in zip(sequence.parameters(), parallel.parameters()):
                v = randn(p1.numel(), device=device)
                assert torch.allclose(p1.hvp(v), p2.hvp(v))


def test_comparison_no_splitting_with_sequence_cpu(num_iters=10):
    """Check if parallel sequence no-split equals HBPSequential on CPU."""
    device = torch.device('cpu')
    compare_no_splitting_with_sequence(device, num_iters)


def test_comparison_no_splitting_with_composition_gpu(num_iters=10):
    """Check if parallel sequence no-split equals HBPSequential on GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        compare_no_splitting_with_sequence(device, num_iters)
    else:
        warn('Could not find CUDA device')


def compare_optimization_no_splitting_with_sequence(device, num_iters):
    """Check if parallel sequence with splitting of 1 behaves
    exactly the same as a normal sequence over multiple
    iterations of CGNewton optimization.
    """
    sequence = example_sequence().to(device)
    parallel = example_sequence_parallel(max_blocks=1).to(device)

    # check equality of parameters
    assert (len(list(sequence.parameters())) == len(
        list(parallel.parameters())) == 6)
    for p1, p2 in zip(sequence.parameters(), parallel.parameters()):
        assert torch.allclose(p1, p2)

    opt1 = CGNewton(
        sequence.parameters(), 0.1, 0.02, cg_atol=1E-8, cg_tol=1E-1)
    opt2 = CGNewton(
        parallel.parameters(), 0.1, 0.02, cg_atol=1E-8, cg_tol=1E-1)

    # check equality of gradients/hvp over multiple runs
    for i in range(num_iters):
        print(i)
        # 5 samples
        input = randn(5, in_features[0], device=device)
        input.requires_grad = True
        # forward pass checking
        out1, loss1 = forward(sequence, input)
        out2, loss2 = forward(parallel, input)
        assert torch.allclose(out1, out2)
        # gradient checking
        loss1.backward()
        loss2.backward()
        for p1, p2 in zip(sequence.parameters(), parallel.parameters()):
            assert p1.grad is not None and p2.grad is not None
            assert not torch.allclose(p1.grad, zeros_like(p1))
            assert not torch.allclose(p2.grad, zeros_like(p2))
            assert torch.allclose(p1.grad, p2.grad)
        loss_hessian = randn(out_features[-1], out_features[-1], device=device)
        # PSD
        loss_hessian = loss_hessian.t().matmul(loss_hessian)
        # check input Hessians and Hessian-vector products
        mod2nd = 'abs'
        # input Hessian
        in_h1 = sequence.backward_hessian(
            loss_hessian,
            compute_input_hessian=True,
            modify_2nd_order_terms=mod2nd)
        in_h2 = parallel.backward_hessian(
            loss_hessian,
            compute_input_hessian=True,
            modify_2nd_order_terms=mod2nd)
        assert torch.allclose(in_h1, in_h2)
        # parameter Hessians
        for p1, p2 in zip(sequence.parameters(), parallel.parameters()):
            v = randn(p1.numel(), device=device)
            assert torch.allclose(p1.hvp(v), p2.hvp(v))

        opt1.step()
        opt2.step()
        opt1.zero_grad()
        opt2.zero_grad()
        for p1, p2 in zip(sequence.parameters(), parallel.parameters()):
            assert p1.grad is not None and p2.grad is not None
            assert torch.allclose(p1.grad, zeros_like(p1))
            assert torch.allclose(p2.grad, zeros_like(p2))
            assert torch.allclose(p1.grad, p2.grad)
            assert torch.allclose(p1, p2)


def test_compare_optimization_no_splitting_with_sequence_cpu(num_iters=50):
    """Check if parallel sequence no-split equals HBPSequential on CPU
    during optimization."""
    device = torch.device('cpu')
    compare_optimization_no_splitting_with_sequence(device, num_iters)


def test_compare_optimization_no_splitting_with_sequence_gpu(num_iters=50):
    """Check if parallel sequence no-split equals HBPSequential on GPU
    during optimization."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        compare_optimization_no_splitting_with_sequence(device, num_iters)
    else:
        warn('Could not find CUDA device')
