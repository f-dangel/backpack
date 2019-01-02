"""Test conversion/splitting HBPCompositionActivation to parallel."""

from torch import randn, Tensor
from ..combined_relu_test import test_input_hessian\
        as hbp_relulinear_after_hessian_backward
from ..combined_sigmoid_test import test_input_hessian\
        as hbp_sigmoidlinear_after_hessian_backward
from ..combined_relu import HBPReLULinear
from ..combined_sigmoid import HBPSigmoidLinear
from .identical import HBPParallelIdentical
from ...utils import (torch_allclose,
                      set_seeds)


in_features = 20
out_features_list = [2, 3, 4]
out_features_list2 = [1, 8]
num_layers = len(out_features_list)
input = randn(1, in_features)


def random_input():
    """Return random input copy."""
    return input.clone()


def test_forward_pass_hbp_relulinear():
    """Test whether single/split modules are consistent in forward."""
    relu_linear = HBPReLULinear(in_features=in_features,
                                out_features=sum(out_features_list),
                                bias=True)
    x = random_input()
    parallel = HBPParallelIdentical.from_module(relu_linear)
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
    parallel = HBPParallelIdentical.from_module(sigma_linear)
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


def test_buffer_copy_hbp_relulinear_unite():
    """Check if buffers are correctly copied over when uniting."""
    layer = hbp_relulinear_after_hessian_backward()
    parallel = HBPParallelIdentical.from_module(layer)

    mod = parallel.get_submodule(0)
    assert isinstance(mod.linear.mean_input, Tensor)
    assert isinstance(mod.activation.grad_phi, Tensor)
    assert isinstance(mod.activation.gradgrad_phi, Tensor)
    assert isinstance(mod.activation.grad_output, Tensor)


def test_buffer_copy_hbp_sigmoidlinear_unite():
    """Check if buffers are correctly copied over when uniting."""
    layer = hbp_sigmoidlinear_after_hessian_backward()
    parallel = HBPParallelIdentical.from_module(layer)

    mod = parallel.get_submodule(0)
    assert isinstance(mod.linear.mean_input, Tensor)
    assert isinstance(mod.activation.grad_phi, Tensor)
    assert isinstance(mod.activation.gradgrad_phi, Tensor)
    assert isinstance(mod.activation.grad_output, Tensor)
