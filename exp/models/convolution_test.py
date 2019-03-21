"""Test the convolutional neural network architectures for the experiments."""

from torch import rand
from torch.nn import (Module, Sigmoid, Linear, Conv2d)
from .convolution import simplest_mnist_cnn
from bpexts.utils import torch_allclose, set_seeds


def test_simplest_mnist_cnn_output_size():
    """Check the output size of the simple CNN on MNIST

    For each batch sample, the output is a 10-dimensional vector.
    """
    x = rand(5, 1, 28, 28)
    model = simplest_mnist_cnn()
    out = model(x)
    assert tuple(out.size()) == (5, 10)


def torch_simplest_mnist_cnn(seed=None):
    """A pure torch implementation of the simple CNN."""

    class SimplestCNN(Module):
        """PyTorch implementation of the simple CNN."""

        def __init__(self):
            super().__init__()
            self.conv = Conv2d(
                in_channels=1, out_channels=8, kernel_size=(4, 4), stride=1)
            self.sigmoid = Sigmoid()
            self.fc = Linear(
                in_features=8 * 25 * 25, out_features=10, bias=True)

        def forward(self, input):
            x = self.conv(input)
            x = self.sigmoid(x)
            # reshape into 2d tensor for fully-connected layer
            x = x.reshape(x.size()[0], -1)
            return self.fc(x)

    # create the model
    set_seeds(seed)
    return SimplestCNN()


def test_compare_simplest_mnist_cnn_parameters():
    """Compare the parameters of the HBP and PyTorch simplest MNIST CNNs."""
    hbp_cnn = simplest_mnist_cnn(seed=1)
    torch_cnn = torch_simplest_mnist_cnn(seed=1)
    assert len(list(hbp_cnn.parameters())) == len(list(torch_cnn.parameters()))
    for p1, p2 in zip(hbp_cnn.parameters(), torch_cnn.parameters()):
        assert torch_allclose(p1, p2)


def test_simplest_mnist_cnn_forward():
    """Compare forward pass of HBP model and PyTorch implementation."""
    hbp_cnn = simplest_mnist_cnn(seed=0)
    torch_cnn = torch_simplest_mnist_cnn(seed=0)
    x = rand(12, 1, 28, 28)
    out_torch = torch_cnn(x)
    out_hbp = hbp_cnn(x)
    assert torch_allclose(out_torch, out_hbp)
