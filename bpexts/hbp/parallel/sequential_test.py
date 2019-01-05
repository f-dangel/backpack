"""Test Hessian backpropagation of sequence of identical parallel modules"""

from torch import randn, eye
from .sequential import HBPParallelSequential
from ..sequential import HBPSequential
from ..linear import HBPLinear
from .linear import HBPParallelLinear
from ..combined_sigmoid import HBPSigmoidLinear
from ..combined_relu import HBPReLULinear
from ..sigmoid import HBPSigmoid
from ..relu import HBPReLU
from .combined import HBPParallelCompositionActivationLinear
from ...utils import (torch_allclose,
                      set_seeds,
                      memory_report)
from ...hessian import exact

in_features = [40, 30, 20, 10, 5]
out_features = [30, 20, 10, 5, 2]
# DO NOT MODIFY!
classes = [
        HBPLinear,
        HBPSigmoidLinear,
        HBPReLULinear,
        HBPSigmoid,
        HBPReLU
        ]
# DO NOT MODIFY!
target_classes = [
        HBPParallelLinear,
        HBPParallelCompositionActivationLinear,
        HBPParallelCompositionActivationLinear,
        HBPSigmoid,
        HBPReLU
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
    for cls, in_, out in zip(classes[:3], in_features[:3], out_features[:3]):
        layers.append(cls(in_features=in_,
                          out_features=out,
                          bias=True))
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
    assert torch_allclose(sequence(x), parallel(x))


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
            return exact.exact_hessian(loss,
                                       [list(parallel.children())[layer_idx]
                                        ._get_parallel_module(parallel_idx)
                                        .weight])
        else:
             return exact.exact_hessian(loss,
                                       [list(parallel.children())[layer_idx]
                                        ._get_parallel_module(parallel_idx)
                                        .linear.weight])
    elif which == 'bias':
        if layer_idx == 0:
            return exact.exact_hessian(loss,
                                       [list(parallel.children())[layer_idx]
                                        ._get_parallel_module(parallel_idx)
                                        .bias])
        else:
             return exact.exact_hessian(loss,
                                       [list(parallel.children())[layer_idx]
                                        ._get_parallel_module(parallel_idx)
                                        .linear.bias])
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
    assert torch_allclose(in_h, brute_force_input_hessian())


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
            w_hessian = linear.weight.hessian()
            w_brute_force = brute_force_hessian(idx, n, 'weight')
            assert torch_allclose(w_hessian, w_brute_force, atol=1E-5)
            # check weight Hessian-vector product
            for _ in range(random_vp):
                v = randn(linear.weight.numel())
                vp = linear.weight.hvp(v)
                vp_result = w_brute_force.matmul(v)
                assert torch_allclose(vp, vp_result, atol=1E-5)


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



def example_loss(tensor):
    """Sum all square entries of a tensor."""
    return (tensor**2).view(-1).sum(0)


def forward(layer, input):
    """Feed input through layers and loss. Return output and loss."""
    out = layer(input)
    return out, example_loss(out)


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
