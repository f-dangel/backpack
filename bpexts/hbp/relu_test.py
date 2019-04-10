"""Test Hessian backpropagation for ReLU layer."""

from torch import randn, tensor
from torch.autograd import grad
from .relu import HBPReLU
from .loss import batch_summed_hessian
from ..utils import torch_allclose


def example_loss(tensor):
    """Sum squared entries of a tensor."""
    return (tensor**2).view(-1).sum(0)


def test_relu_derivatives():
    """Test first and second derivative of ReLU."""
    layer = HBPReLU()
    x = tensor([[1., 0., -1.], [2., 5., -3.]])
    # forward, calls hook
    out = layer(x)
    relu_x = tensor([[1., 0., 0.], [2., 5., 0.]])
    assert torch_allclose(out, relu_x)
    # check first derivative
    grad_relu = tensor([[1., 0., 0.], [1., 1., 0.]])
    assert torch_allclose(grad_relu, layer.grad_phi)
    # check second derivative
    gradgrad_relu = tensor([[0., 0., 0.], [0., 0., 0.]])
    assert torch_allclose(gradgrad_relu, layer.gradgrad_phi)


def layer_with_input_output_and_loss():
    """Return layer with input, output and loss."""
    layer = HBPReLU()
    x = tensor([[-1., 2., -3.]], requires_grad=True)
    out = layer(x)
    loss = example_loss(out)
    return layer, x, out, loss


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
    x_hessian = tensor([[0., 0., 0.], [0., 2., 0.], [0., 0., 0.]])
    assert torch_allclose(x_hessian, input_hessian)
