"""Test batch gradient computation of linear layer."""

from torch import Tensor
from torch.nn import Linear
from .linear import G_Linear
from ..utils import torch_allclose


# predefined weight matrix and bias
weight = Tensor([[1, 2, 3],
                 [4, 5, 6]]).float()
bias = Tensor([7, 8]).float()
in_features, out_features = 3, 2

# linaer layer with fixed weight and bias
lin = Linear(in_features=in_features, out_features=out_features)
lin.weight.data = weight
lin.bias.data = bias

# extended linear layer with fixed weight and bias
g_lin = G_Linear(in_features=in_features, out_features=out_features)
g_lin.weight.data = weight
g_lin.bias.data = bias


def loss_function(tensor):
    """Test loss function. Sum over squared entries."""
    return ((tensor.view(-1))**2).sum()


# input (1 sample)
in1 = Tensor([1, 1, 1]).float()
out1 = Tensor([6 + 7, 15 + 8]).float()
loss1 = 13**2 + 23**2
bias_grad1 = Tensor([2 * 13, 2 * 23]).float()
bias_grad_batch1 = bias_grad1

# input (2 samples)
in2 = Tensor([[1, 0, 1],
              [0, 1, 0]]).float()
out2 = Tensor([[4 + 7, 10 + 8],
               [2 + 7, 5 + 8]]).float()
loss2 = 11**2 + 18**2 + 9**2 + 13**2
bias_grad2 = Tensor([2 * (11 + 9), 2 * (18 + 13)])
bias_grad_batch2 = Tensor([[2 * 11, 2 * 18],
                           [2 * 9, 2 * 13]]).float()


# inputs, results, losses, bias_grads and bias_batch_grads
inputs = [in1, in2]
results = [out1, out2]
losses = [loss1, loss2]
bias_grads = [bias_grad1, bias_grad2]
bias_grads_batch = [bias_grad_batch1, bias_grad_batch2]


def test_forward():
    """Compare forward of torch.nn.Linear and exts.gradient.G_Linear.

    Handles single-instance and batch mode."""
    for input, result in zip(inputs, results):
        out_lin = lin(input)
        out_g_lin = g_lin(input)
        assert torch_allclose(out_lin, out_g_lin)


def test_losses():
    """Test output of loss function."""
    for input, loss in zip(inputs, losses):
        out = g_lin(input)
        loss_val = loss_function(out)
        assert loss_val.item() == loss


def test_bias_grad():
    """Test computation of bias gradients."""
    for input, bias_grad in zip(inputs, bias_grads):
        out = g_lin(input)
        loss = loss_function(out)
        loss.backward()
        assert torch_allclose(g_lin.bias.grad, bias_grad)
        g_lin.zero_grad()


def test_bias_grad_batch():
    """Test computation of bias batch gradients."""
    for input, bias_grad_batch in zip(inputs, bias_grads_batch):
        out = g_lin(input)
        loss = loss_function(out)
        loss.backward()
        assert torch_allclose(g_lin.bias.grad_batch, bias_grad_batch)
        g_lin.zero_grad()
