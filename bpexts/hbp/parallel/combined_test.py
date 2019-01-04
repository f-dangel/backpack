"""Test conversion/splitting HBPCompositionActivation to parallel."""

from torch import randn, Tensor, eye
from ..combined_relu_test import test_input_hessian\
        as hbp_relulinear_after_hessian_backward
from ..combined_sigmoid_test import test_input_hessian\
        as hbp_sigmoidlinear_after_hessian_backward
from ..combined_relu import HBPReLULinear
from ..combined_sigmoid import HBPSigmoidLinear
from .combined import HBPParallelCompositionActivationLinear
from ...utils import (torch_allclose,
                      set_seeds,
                      memory_report)


in_features = 20
out_features_list = [2, 3, 4]
out_features_list2 = [1, 8]
num_layers = len(out_features_list)
input = randn(1, in_features)


def test_forward_pass_hbp_relulinear():
    """Test whether single/split modules are consistent in forward."""
    relu_linear = HBPReLULinear(in_features=in_features,
                                out_features=sum(out_features_list),
                                bias=True)
    x = random_input()
    parallel = HBPParallelCompositionActivationLinear(relu_linear)
    assert torch_allclose(relu_linear(x), parallel(x))

    parallel2 = parallel.split(out_features_list)
    assert torch_allclose(relu_linear(x), parallel2(x))

    parallel3 = parallel2.unite()
    assert torch_allclose(relu_linear.linear.bias,
                          parallel3.get_submodule(0).linear.bias)
    assert torch_allclose(relu_linear.linear.weight,
                          parallel3.get_submodule(0).linear.weight)
    assert torch_allclose(relu_linear(x), parallel3(x))

    parallel4 = parallel2.split(out_features_list2)
    assert torch_allclose(relu_linear(x), parallel4(x))


def test_forward_pass_hbp_sigmoidlinear():
    """Test whether single/split modules are consistent in forward."""
    sigma_linear = HBPSigmoidLinear(in_features=in_features,
                                    out_features=sum(out_features_list),
                                    bias=True)
    x = random_input()
    parallel = HBPParallelCompositionActivationLinear(sigma_linear)
    assert torch_allclose(sigma_linear(x), parallel(x))

    parallel2 = parallel.split(out_features_list)
    assert torch_allclose(sigma_linear(x), parallel2(x))

    parallel3 = parallel2.unite()
    assert torch_allclose(sigma_linear.linear.bias,
                          parallel3.get_submodule(0).linear.bias)
    assert torch_allclose(sigma_linear.linear.weight,
                          parallel3.get_submodule(0).linear.weight)
    assert torch_allclose(sigma_linear(x), parallel3(x))

    parallel4 = parallel2.split(out_features_list2)
    assert torch_allclose(sigma_linear(x), parallel4(x))


def test_buffer_split_hbp_relulinear_unite():
    """Check if buffers are correctly copied over when uniting."""
    layer = hbp_relulinear_after_hessian_backward()
    parallel = HBPParallelCompositionActivationLinear(layer)

    mod = parallel.get_submodule(0)
    assert isinstance(mod.linear.mean_input, Tensor)
    assert isinstance(mod.activation.grad_phi, Tensor)
    assert isinstance(mod.activation.gradgrad_phi, Tensor)
    assert isinstance(mod.activation.grad_output, Tensor)

    parallel2 = parallel.split_into_blocks(2)
    for i, mod in enumerate(parallel2.children()):
        assert isinstance(mod.linear.mean_input, Tensor)
        if i == 0:
            assert hasattr(mod.activation, 'grad_phi')
            assert hasattr(mod.activation, 'gradgrad_phi')
            assert hasattr(mod.activation, 'grad_output')
        else:
            assert not hasattr(mod.activation, 'grad_phi')
            assert not hasattr(mod.activation, 'gradgrad_phi')
            assert not hasattr(mod.activation, 'grad_output')
 
    parallel3 = parallel.unite()
    for i, mod in enumerate(parallel3.children()):
        assert isinstance(mod.linear.mean_input, Tensor)
        if i == 0:
            assert hasattr(mod.activation, 'grad_phi')
            assert hasattr(mod.activation, 'gradgrad_phi')
            assert hasattr(mod.activation, 'grad_output')
        else:
            assert not hasattr(mod.activation, 'grad_phi')
            assert not hasattr(mod.activation, 'gradgrad_phi')
            assert not hasattr(mod.activation, 'grad_output')


def test_buffer_split_hbp_sigmoidlinear_unite():
    """Check if buffers are correctly copied over when uniting."""
    layer = hbp_sigmoidlinear_after_hessian_backward()
    parallel = HBPParallelCompositionActivationLinear(layer)

    mod = parallel.get_submodule(0)
    assert isinstance(mod.linear.mean_input, Tensor)
    assert isinstance(mod.activation.grad_phi, Tensor)
    assert isinstance(mod.activation.gradgrad_phi, Tensor)
    assert isinstance(mod.activation.grad_output, Tensor)

    parallel2 = parallel.split_into_blocks(2)
    for i, mod in enumerate(parallel2.children()):
        assert isinstance(mod.linear.mean_input, Tensor)
        if i == 0:
            assert hasattr(mod.activation, 'grad_phi')
            assert hasattr(mod.activation, 'gradgrad_phi')
            assert hasattr(mod.activation, 'grad_output')
        else:
            assert not hasattr(mod.activation, 'grad_phi')
            assert not hasattr(mod.activation, 'gradgrad_phi')
            assert not hasattr(mod.activation, 'grad_output')
 
    parallel3 = parallel.unite()
    for i, mod in enumerate(parallel3.children()):
        assert isinstance(mod.linear.mean_input, Tensor)
        if i == 0:
            assert hasattr(mod.activation, 'grad_phi')
            assert hasattr(mod.activation, 'gradgrad_phi')
            assert hasattr(mod.activation, 'grad_output')
        else:
            assert not hasattr(mod.activation, 'grad_phi')
            assert not hasattr(mod.activation, 'gradgrad_phi')
            assert not hasattr(mod.activation, 'grad_output')


def test_memory_consumption():
    """Check memory consumption during splitting."""
    layer = hbp_sigmoidlinear_after_hessian_backward()
    # delete Hessian quantities
    del layer.linear.bias.hessian
    del layer.linear.weight.hessian
    del layer.linear.weight.hvp
    print('Before conversion')
    mem_stat1 = memory_report()

    layer = HBPParallelCompositionActivationLinear(layer)
    print('After conversion')
    mem_stat2 = memory_report()

    layer = layer.split_into_blocks(4)
    print('After splitting')
    mem_stat3 = memory_report()
    assert mem_stat1 == mem_stat2 == mem_stat3


def random_input():
    """Return random input copy."""
    new_input = input.clone()
    new_input.requires_grad = True
    return new_input


def example_loss(tensor):
    """Sum all square entries of a tensor."""
    return (tensor**2).view(-1).sum(0)


def forward(layer, input):
    """Feed input through layers and loss. Return output and loss."""
    out = layer(input)
    return out, example_loss(out)


def test_memory_consumption_during_hbp():
    """Check for constant memory consumption during HBP."""
    sigma_linear = HBPSigmoidLinear(in_features=in_features,
                                    out_features=sum(out_features_list),
                                    bias=True)
    sigma_linear = HBPParallelCompositionActivationLinear(
            sigma_linear).split_into_blocks(4)

    mem_stat = None
    mem_stat_after = None
    
    for i in range(10):
        input = random_input()
        out, loss = forward(sigma_linear, input)
        loss_hessian = 2 * eye(out.numel())
        loss.backward()
        out_h = loss_hessian
        sigma_linear.backward_hessian(out_h)
        if i == 0:
            mem_stat = memory_report()
        if i == 9:
            mem_stat_after = memory_report()

    assert mem_stat == mem_stat_after
