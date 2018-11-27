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
in1 = Tensor([[1, 1, 1]]).float()
out1 = Tensor([[6 + 7, 15 + 8]]).float()
loss1 = 13**2 + 23**2
bias_grad1 = Tensor([2 * 13, 2 * 23]).float()
bias_grad_batch1 = bias_grad1
weight_grad1 = Tensor([[26, 26, 26],
                       [46, 46, 46]]).float()
weight_grad_batch1 = weight_grad1

# input (2 samples)
in2 = Tensor([[1, 0, 1],
              [0, 1, 0]]).float()
out2 = Tensor([[4 + 7, 10 + 8],
               [2 + 7, 5 + 8]]).float()
loss2 = 11**2 + 18**2 + 9**2 + 13**2
bias_grad2 = Tensor([2 * (11 + 9), 2 * (18 + 13)])
bias_grad_batch2 = Tensor([[2 * 11, 2 * 18],
                           [2 * 9, 2 * 13]]).float()
weight_grad2 = Tensor([[22, 18, 22],
                       [36, 26, 36]]).float()
weight_grad_batch2 = Tensor([[[22, 0, 22],
                              [36, 0, 36]],
                             [[0, 18, 0],
                              [0, 26, 0]]]).float()

# as lists for zipping
inputs = [in1, in2]
results = [out1, out2]
losses = [loss1, loss2]
bias_grads = [bias_grad1, bias_grad2]
bias_grads_batch = [bias_grad_batch1, bias_grad_batch2]
weight_grads = [weight_grad1, weight_grad2]
weight_grads_batch = [weight_grad_batch1, weight_grad_batch2]


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


def test_grad():
    """Test computation of bias/weight gradients."""
    for input, b_grad, w_grad in zip(inputs,
                                     bias_grads,
                                     weight_grads):
        out = g_lin(input)
        loss = loss_function(out)
        loss.backward()
        assert torch_allclose(g_lin.bias.grad, b_grad)
        assert torch_allclose(g_lin.weight.grad, w_grad)
        g_lin.zero_grad()
        g_lin.clear_grad_batch()
        g_lin.remove_exts_buffers()


def test_grad_batch():
    """Test computation of bias/weight batch gradients."""
    for input, b_grad_batch, w_grad_batch in zip(inputs,
                                                 bias_grads_batch,
                                                 weight_grads_batch):
        out = g_lin(input)
        loss = loss_function(out)
        loss.backward()
        assert torch_allclose(g_lin.bias.grad_batch, b_grad_batch)
        print(g_lin.weight.grad_batch)
        print(w_grad_batch)
        assert torch_allclose(g_lin.weight.grad_batch, w_grad_batch)
        g_lin.zero_grad()
        g_lin.clear_grad_batch()
        g_lin.remove_exts_buffers()
