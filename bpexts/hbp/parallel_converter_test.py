"""Test Hessian backpropagation of shared linear layer."""

from torch import randn, eye
from .parallel_converter import HBPParallelModule
from .linear import HBPLinear
from .combined_sigmoid import HBPSigmoidLinear
from ..utils import (torch_allclose,
                     set_seeds)
from ..hessian import exact

in_features = 20
out_features_list = [2, 3, 4]
num_layers = len(out_features_list)
input = randn(1, in_features)


def random_input():
    """Return random input copy."""
    return input.clone()


def create_layer():
    """Return example linear layer."""
    # same seed
    set_seeds(0)
    return HBPParallelModule.hbp_linear_with_splitting(
            in_features=in_features,
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


def brute_force_input_hessian():
    """Compute the Hessian with respect to the input by brute force."""
    layer = create_layer()
    input = random_input()
    input.requires_grad = True
    _, loss = forward(layer, input)
    return exact.exact_hessian(loss, [input])


def test_input_hessians():
    """Test whether Hessian with respect to input is correctly reproduced."""
    layer = create_layer()
    out, loss = forward(layer, random_input())
    loss_hessian = 2 * eye(out.numel())
    loss.backward()
    # call HBP recursively
    in_h = layer.backward_hessian(loss_hessian)
    assert torch_allclose(in_h, brute_force_input_hessian())


def test_forward_pass_hbp_linear():
    """Test whether single module = parallel modules in forward mode."""
    linear = HBPLinear(in_features=in_features,
                       out_features=sum(out_features_list),
                       bias=True)
    x = random_input()
    parallel = HBPParallelModule.fromHBPLinear(linear, out_features_list)
    assert torch_allclose(linear(x), parallel(x))


def test_forward_pass_hbp_sigmoidlinear():
    """Test whether single module = parallel modules in forward mode."""
    sigmoid_linear = HBPSigmoidLinear(in_features=in_features,
                                      out_features=sum(out_features_list),
                                      bias=True)
    x = random_input()
    parallel = HBPParallelModule.fromHBPCompositionActivationLinear(
            sigmoid_linear,
            out_features_list)
    assert torch_allclose(sigmoid_linear(x), parallel(x))
