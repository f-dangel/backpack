r"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import quickbench
from load_mnist import MNISTLoader


class CNN(nn.Module):

    def __init__(self, modules):
        r"""
        CNN for MNIST
        """
        conv = modules['conv']
        linear = modules['linear']

        super(CNN, self).__init__()

        self.conv1 = conv(1, 64, 5, 2)
        self.conv2 = conv(64, 96, 3, 2)
        self.conv3 = conv(96, 128, 3, 2)
        self.fc1 = linear(512, 512)
        self.fc2 = linear(512, 256)
        self.fc3 = linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view((-1, 512))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __repr__(self):
        return "CNN(conv[1, 64, 5, 2], conv[64, 96, 3, 2], conv[96, 128, 3, 2], fc[512, 512], fc[512, 256], fc[256, 10])"


class MLP(nn.Sequential):

    def __init__(self, L, modules):
        r"""
        Creates a MultilayerPerception for MNIST,
        with a given implementation of linear layers.

        L is the number of intermediate layers; the architecture is
        [784 x 256] -> ([256 x 256])*L -> [256 x 10]
        """
        linear = modules['linear']

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
        return "MLP(" + ", ".join(["[%d, %d]" % (l.in_features, l.out_features) for l in self.children()]) + ")"


def run(make_model, functions, modules, N, correctness=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    nnModel = make_model(modules['nn']).to(device)

    torch.manual_seed(0)
    bpextsModel = make_model(modules['bpexts']).to(device)

    mnist_loader = MNISTLoader(train_batch_size=N).train_loader()
    x, y = next(iter(mnist_loader))
    x, y = x.to(device), y.to(device)

    def datafunc():
        return x, y, nnModel, bpextsModel

    if correctness:
        quickbench.check(datafunc, [functions['nn'], functions['bpexts']],
                         compfunc=lambda x, y: all([torch.allclose(i, j, atol=10e-5) for (i, j) in zip(x, y)]))

    timings = quickbench.bench(datafunc, [functions['nn'], functions['bpexts']], reps=20, prettyprint=False)

    print(("%d\t" + "%.2fs\t" + "%.2fs\t" + "~%.1fx") %
          (N, timings[0], timings[1], timings[0] / timings[1]))
