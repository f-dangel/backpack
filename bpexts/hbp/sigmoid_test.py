"""Test Hessian backpropagation for Sigmoid layer."""

from torch import randn, tensor, sigmoid
from torch.autograd import grad
from .sigmoid import HBPSigmoid
from .loss import batch_summed_hessian
from ..utils import torch_allclose
from ..hessian.exact import exact_hessian


def example_loss(tensor):
    """Sum squared entries of a tensor."""
    return (tensor ** 2).view(-1).sum(0)


def test_sigmoid_derivatives():
    """Test first and second derivative of sigmoid."""
    layer, x = HBPSigmoid(), randn(2, 3, 4)
    # forward, calls hook
    out = layer(x)
    sigma = sigmoid(x)
    assert torch_allclose(out, sigma)
    # check first derivative
    grad_sigma = sigma * (1 - sigma)
    assert torch_allclose(grad_sigma, layer.grad_phi)
    # check second derivative
    gradgrad_sigma = sigma * (1 - sigma) * (1 - 2 * sigma)
    assert torch_allclose(gradgrad_sigma, layer.gradgrad_phi)


def layer_with_input_output_and_loss():
    """Return layer with input, output and loss."""
    layer = HBPSigmoid()
    x = tensor([[1, 2, 3]], requires_grad=True).float()
    out = layer(x)
    loss = example_loss(out)
    return layer, x, out, loss


def layer_input_hessian():
    """Compute the Hessian with respect to the input."""
    layer, x, out, loss = layer_with_input_output_and_loss()
    input_hessian = exact_hessian(loss, [x])
    return input_hessian


def test_sigmoid_grad_output():
    """Test storing of gradient with respect to layer output."""
    layer, x, out, loss = layer_with_input_output_and_loss()
    # result for comparison
    out_grad, = grad(loss, out, create_graph=True)
    # call backward to trigger backward hook
    loss.backward()
    assert torch_allclose(out_grad, layer.grad_output)


def test_sigmoid_input_hessian():
    """Test Hessian backpropagation for sigmoid."""
    layer, x, out, loss = layer_with_input_output_and_loss()
    # important note: compute loss Hessian before calling backward
    loss_hessian = batch_summed_hessian(loss, out)
    loss.backward()
    input_hessian = layer.backward_hessian(loss_hessian)
    print(input_hessian)
    x_hessian = layer_input_hessian()
    assert torch_allclose(x_hessian, input_hessian)
