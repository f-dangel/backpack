r"""
Script to run some timing benchmarks on the batch gradient computation
for linear and convolution modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import helpers
import quickbench
from bpexts.gradient.linear import G_Linear
from bpexts.gradient.conv2d import G_Conv2d

###
# Models
###


class CNN(nn.Sequential):

    def __init__(self, L, batch=False):
        r"""
        Creates a MultilayerPerception for MNIST,
        with a given implementation of linear layers.

        L is the number of intermediate layers; the architecture is
        [784 x 256] -> ([256 x 256])*L -> [256 x 10]
        """
        conv = nn.Conv2d if batch is False else G_Conv2d
        linear = nn.Linear if batch is False else G_Linear

        layers = []
        layers.append(conv(1, 32, 3, 1))
        for l in range(L):
            layers.append(conv(32, 32, 3, 1))
        self.outD = int(32 * (28 - 2 * (L + 1))**2)
        layers.append(linear(self.outD, 10))

        super(CNN, self).__init__(*layers)

    def forward(self, x):
        layers = list(self.children())
        for l in layers[0:-1]:
            x = F.relu(l(x))
        x = x.view((-1, self.outD))
        x = layers[-1](x)

        return x

    def __repr__(self):
        return "CNN(\n" + "\n".join([str(l) for l in self.children()]) + "\n)"


class MLP(nn.Sequential):

    def __init__(self, L, batch=False):
        r"""
        Creates a MultilayerPerception for MNIST,
        with a given implementation of linear layers.

        L is the number of intermediate layers; the architecture is
        [784 x 256] -> ([256 x 256])*L -> [256 x 10]
        """
        linear = nn.Linear if batch is False else G_Linear

        layers = []
        layers.append(linear(784, 256))
        for l in range(L):
            layers.append(linear(256, 256))
        layers.append(linear(256, 10))

        super(MLP, self).__init__(*layers)

    def forward(self, x):
        x = x.view(-1, 784)
        layers = list(self.children())
        for l in layers[0:-1]:
            x = F.relu(l(x))
        x = layers[-1](x)
        return x

    def __repr__(self):
        return "MLP(\n" + "\n".join([str(l) for l in self.children()]) + "\n)"


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

    return batch_grads


def batchGrad(x, y, _, model):
    r"""
    Computes individual gradient for the minibatch (x,y) using batch autograd
    """
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return list([p.grad_batch for p in model.parameters()])


###
# Benchmarking helpers
###


def benchmark_for(L, N, make_model):
    r"""
    Run a benchmark with a minibatch of size N and a L+2 Layers MLP
    """
    torch.manual_seed(0)
    mnist_loader = helpers.load_MNIST(batch_size=N, use_cuda=False)
    x, y = next(iter(mnist_loader))

    forloopModel = make_model(L, batch=False)
    batchModel = make_model(L, batch=True)

    def datafunc():
        return x, y, forloopModel, batchModel

    return quickbench.bench(datafunc, [forloop, batchGrad], reps=1, prettyprint=False)


if __name__ == "__main__":

    def run(model):
        print("L\t" + "N\t" + "ForLoop\t" + "Batch\t" + "Improv.")
        for L in [0, 1, 2]:
            for N in [128, 512, 1024]:
                timings = benchmark_for(L, N, model)
                print(("%d\t" + "%d\t" + "%.2fs\t" + "%.2fs\t" + "~%.1fx") %
                      (L, N, timings[0], timings[1], timings[0] / timings[1]))

    print("MLP:")
    run(MLP)
    print("CNN:")
    run(CNN)
