r"""
Script to run some timing benchmarks on the batch gradient computation
for linear and convolution modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from bpexts.sumgradsquared.linear import SumGradSquared_Linear as SGS_Linear
from bpexts.sumgradsquared.conv2d import SumGradSquared_Conv2d as SGS_Conv2d
from bench_helpers import CNN, MLP, run

###
# Batch Gradient Computation
###


def forloop(x, y, model, _):
    r"""
    Computes individual gradient for the minibatch (x,y) using a for loop
    """
    def init_batch_gradients():
        batch_grads = []
        for param in model.parameters():
            batch_grads.append(torch.zeros(x.shape[0], *param.shape))
        return batch_grads

    def compute_grad(x, y):
        loss = F.cross_entropy(model(x), y)
        return torch.autograd.grad(loss, model.parameters())

    batch_grads = init_batch_gradients()
    for n in range(x.shape[0]):
        grads = compute_grad(x[n, :].unsqueeze(0), y[n].unsqueeze(0))
        for i, g in enumerate(grads):
            batch_grads[i][n, :] = g

    for i, g in enumerate(batch_grads):
        batch_grads[i] = torch.sum(g**2, dim=0)

    return batch_grads


def batchGrad(x, y, _, model):
    r"""
    Computes individual gradient for the minibatch (x,y) using batch autograd
    """
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    #print(list(model.parameters())[1].grad_batch* x.shape[0])
    return list([p.sum_grad_squared * x.shape[0]**2 for p in model.parameters()])


if __name__ == "__main__":

    modules = {
        'nn': {
            'linear': nn.Linear,
            'conv': nn.Conv2d,
        },
        'bpexts': {
            'linear': SGS_Linear,
            'conv': SGS_Conv2d,
        }
    }

    functions = {
        'nn': forloop,
        'bpexts': batchGrad
    }

    print("N\t" + "Naive\t" + "BPexts\t" + "Ratio")
    print("-----\t" + "-----\t" + "-----\t" + "------")

    Ns = [1, 8, 64]
    print(MLP(0, modules['nn']))
    for N in Ns:
        run(make_model=lambda modules: MLP(0, modules), functions=functions, modules=modules, N=N)

    print(MLP(1, modules['nn']))
    for N in Ns:
        run(make_model=lambda modules: MLP(1, modules), functions=functions, modules=modules, N=N)

    print(MLP(2, modules['nn']))
    for N in Ns:
        run(make_model=lambda modules: MLP(2, modules), functions=functions, modules=modules, N=N)

    print(CNN(modules['nn']))
    for N in Ns:
        run(make_model=CNN, functions=functions, modules=modules, N=N)
