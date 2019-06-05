"""Test HBP of sigmoid layer."""

import torch
import unittest
from torch.nn import ReLU
from bpexts.hbp.relu import HBPReLU
from .hbp_test import set_up_hbp_tests

# hyper-parameters
in_features = 30
bias = True
input_size = (1, in_features)
atol = 5e-6
rtol = 1e-5
num_hvp = 10


def torch_fn():
    """Create a ReLU layer in torch."""
    return ReLU()


def hbp_fn():
    """Create a ReLU layer with HBP functionality."""
    return HBPReLU()


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_fn,
        'HBPReLU',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


def hbp_from_torch_fn():
    """Create HBPReLU from ReLU."""
    torch_layer = torch_fn()
    return HBPReLU.from_torch(torch_layer)


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_from_torch_fn,
        'HBPReLUFromTorch',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


class HBPReLUAdditionalTests(unittest.TestCase):
    @staticmethod
    def example_loss(tensor):
        """Sum squared entries of a tensor."""
        return (tensor**2).view(-1).sum(0)

    def test_relu_derivatives(self):
        """Test first and second derivative of ReLU."""
        layer = HBPReLU()
        x = torch.tensor([[1., 0., -1.], [2., 5., -3.]])
        # forward, calls hook
        out = layer(x)
        relu_x = torch.tensor([[1., 0., 0.], [2., 5., 0.]])
        assert torch.allclose(out, relu_x)
        # check first derivative
        grad_relu = torch.tensor([[1., 0., 0.], [1., 1., 0.]])
        assert torch.allclose(grad_relu, layer.grad_phi)
        # check second derivative
        gradgrad_relu = torch.tensor([[0., 0., 0.], [0., 0., 0.]])
        assert torch.allclose(gradgrad_relu, layer.gradgrad_phi)

    def layer_with_input_output_and_loss(self):
        """Return layer with input, output and loss."""
        layer = HBPReLU()
        x = torch.tensor([[-1., 2., -3.]], requires_grad=True)
        out = layer(x)
        loss = self.example_loss(out)
        return layer, x, out, loss

    def test_relu_grad_output(self):
        """Test storing of gradient with respect to layer output."""
        layer, x, out, loss = self.layer_with_input_output_and_loss()
        # result for comparison
        out_grad, = torch.autograd.grad(loss, out, create_graph=True)
        # call backward to trigger backward hook
        loss.backward()
        assert torch.allclose(out_grad, layer.grad_output)
