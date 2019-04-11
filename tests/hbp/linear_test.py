"""Test HBP of linear layer."""

import unittest
import torch
from torch.nn import Linear
from bpexts.hbp.linear import HBPLinear
from bpexts.hbp.loss import batch_summed_hessian
from bpexts.utils import set_seeds
from .hbp_test import set_up_hbp_tests

# hyper-parameters
in_features = 50
out_features = 10
bias = True
input_size = (1, in_features)
atol = 5e-5
rtol = 1e-5
num_hvp = 10


def torch_fn():
    """Create a linear layer in torch."""
    set_seeds(0)
    return Linear(
        in_features=in_features, out_features=out_features, bias=bias)


def hbp_fn():
    """Create a linear layer with HBP functionality."""
    set_seeds(0)
    return HBPLinear(
        in_features=in_features, out_features=out_features, bias=bias)


for name, test_cls in set_up_hbp_tests(
        torch_fn,
        hbp_fn,
        'HBPLinear',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls


class HBPLinearHardcodedTest(unittest.TestCase):
    """Hardcoded test of ``HBPLinear``."""

    @staticmethod
    def example_layer():
        """Return example linear layer."""
        layer = HBPLinear(in_features=3, out_features=2)
        w = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        b = torch.tensor([7., 8.], requires_grad=True)
        layer.weight.data = w
        layer.bias.data = b
        return layer

    @staticmethod
    def example_input():
        """Example input for the linear layer."""
        x = torch.tensor([[1., 3., 5.]], requires_grad=True)
        return x

    @staticmethod
    def example_loss(tensor):
        """Sum all square entries of a tensor."""
        return (tensor**2).view(-1).sum(0)

    def test_mean_input_hook(self):
        """Check storing of mean_input hook."""
        layer = self.example_layer()
        x = torch.tensor([[2., 4., 6.], [8., 10., 12.], [14., 16., 18.]])
        mean_x = torch.tensor([[8., 10., 12.]])
        # forward, calls hook
        layer(x)
        assert torch.allclose(layer.mean_input, mean_x)

    def test_input_hessian(self):
        """Return layer after backward_hessian, check input Hessian."""
        layer, x = self.example_layer(), self.example_input()
        out = layer(x)
        loss = self.example_loss(out)
        # Hessian of loss function w.r.t layer output
        output_hessian = batch_summed_hessian(loss, out)
        loss_hessian = torch.tensor([[2., 0.], [0., 2.]])
        assert torch.allclose(loss_hessian, output_hessian)
        # Hessian with respect to layer inputs
        # call hessian backward
        input_hessian = layer.backward_hessian(loss_hessian)
        # result: W^T * output_hessian * W
        h_in_result = torch.tensor([[34., 44., 54.], [44., 58., 72.],
                                    [54., 72., 90.]])
        assert torch.allclose(input_hessian, h_in_result)
        return layer

    def test_bias_hessian(self, random_vp=10):
        """Check correct backpropagation of bias Hessian and HVP."""
        layer = self.test_input_hessian()
        # Hessian with respect to layer bias
        bias_hessian = torch.tensor([[2., 0.], [0., 2.]])
        assert torch.allclose(layer.bias.hessian, bias_hessian)
        # check Hessian-vector product
        for _ in range(random_vp):
            v = torch.randn(2)
            vp = layer.bias.hvp(v)
            result = bias_hessian.matmul(v)
            assert torch.allclose(vp, result, atol=1E-5)

    def test_weight_hessian(self, random_vp=10):
        """Check correct weight Hessian/HVP backpropagation."""
        layer = self.test_input_hessian()
        # Hessian with respect to layer weights
        # x * x^T \otimes output_hessian
        weight_hessian = torch.tensor([[2., 6., 10., 0., 0., 0.],
                                       [6., 18., 30., 0., 0., 0.],
                                       [10., 30., 50., 0., 0., 0.],
                                       [0., 0., 0., 2., 6., 10.],
                                       [0., 0., 0., 6., 18., 30.],
                                       [0., 0., 0., 10., 30., 50.]])
        print(layer.weight.hessian())
        assert torch.allclose(layer.weight.hessian(), weight_hessian)
        # check Hessian-vector product
        for _ in range(random_vp):
            v = torch.randn(6)
            vp = layer.weight.hvp(v)
            result = weight_hessian.matmul(v)
            assert torch.allclose(vp, result, atol=1E-5)
