"""
Tests the computation of sum-of-squared gradients
"""

import torch
from torch.nn import Conv2d
import bpexts.gradient.config as config
from bpexts.gradient.conv2d import Conv2d as SGS_Conv2d

torch.manual_seed(0)
X1 = torch.rand(1, 3, 28, 28)
X2 = torch.rand(2, 3, 28, 28)
X4 = torch.rand(4, 3, 28, 28)
inputs = [X1, X2, X4]

torch.manual_seed(0)
nn_layer = Conv2d(3, 64, 5, 2)
torch.manual_seed(0)
sgs_layer = SGS_Conv2d(3, 64, 5, 2)


def test_forward():
    for X in inputs:
        Y1 = nn_layer(X)
        Y2 = sgs_layer(X)
        assert torch.allclose(Y1, Y2)


def test_backward():
    for X in inputs:
        Y1 = nn_layer(X).norm(2)
        Y1.backward()
        Y2 = sgs_layer(X).norm(2)
        Y2.backward()

        for (p1, p2) in zip(nn_layer.parameters(), sgs_layer.parameters()):
            assert torch.allclose(p1.grad, p2.grad)


def forloopSGS(x, conv):
    batch_grads = []

    # init
    for p in conv.parameters():
        batch_grads.append(torch.zeros(x.shape[0], *p.shape))

    # individual gradients
    for n in range(x.shape[0]):
        loss = conv(x[n, :].unsqueeze(0)).norm(2)**2
        grad = torch.autograd.grad(loss, conv.parameters())

        for i, g in enumerate(grad):
            print(g)
            batch_grads[i][n, :] = g

    # sum-of-squares
    for i, g in enumerate(batch_grads):
        batch_grads[i] = torch.sum(g**2, dim=0)

    return batch_grads


def test_sum_grad_squared():
    for X in inputs:
        baseline = forloopSGS(X, nn_layer)
        with config.bpexts(config.SUM_GRAD_SQUARED):
            (sgs_layer(X).norm(2)**2).backward()
            for (p1, p2) in zip(baseline, sgs_layer.parameters()):
                assert torch.allclose(p1, p2.sum_grad_squared)
            sgs_layer.clear_grad_batch()
            sgs_layer.clear_sum_grad_squared()
