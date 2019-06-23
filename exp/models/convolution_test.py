"""Test the convolutional neural network architectures for the experiments."""

from torch import rand, allclose
from torch.nn import (Module, Sigmoid, Linear, Conv2d)
from .convolution import c1d1
from bpexts.utils import set_seeds
from bpexts.hbp.conv2d import HBPConv2d
from numpy import prod


def test_c1d1_output_size():
    """Check the output size of the simple CNN on MNIST

    For each batch sample, the output is a 10-dimensional vector.
    """
    x = rand(5, 1, 28, 28)
    model = c1d1()
    out = model(x)
    assert tuple(out.size()) == (5, 10)

    x = rand(3, 1, 20, 24)
    model = c1d1(input_size=(1, 20, 24), num_outputs=8)
    out = model(x)
    assert tuple(out.size()) == (3, 8)


def torch_c1d1(
        # dimension of input and output of the net
        input_size=(1, 28, 28),
        num_outputs=10,
        # convolution parameters
        conv_channels=8,
        kernel_size=4,
        padding=0,
        stride=1):
    """Torch implementation of the simple CNN for fixed hyperparameters."""

    class c1d1Torch(Module):
        """PyTorch implementation of c1d1."""

        def __init__(self):
            super().__init__()
            # determine output size of convolution layer
            output_size = HBPConv2d.output_shape(
                input_size=(1, ) + input_size,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                stride=stride)
            output_numel = prod(output_size)
            # create layers
            self.conv = Conv2d(
                in_channels=input_size[0],
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride)
            self.sigmoid = Sigmoid()
            self.fc = Linear(
                in_features=output_numel, out_features=num_outputs, bias=True)

        def forward(self, input):
            x = self.conv(input)
            x = self.sigmoid(x)
            # flatten for fully-connected layer
            x = x.reshape(x.size()[0], -1)
            return self.fc(x)

    return c1d1Torch()


def test_compare_c1d1_parameters():
    """Compare the parameters of the HBP and PyTorch c1d1."""
    set_seeds(0)
    hbp_cnn = c1d1()
    set_seeds(0)
    torch_cnn = torch_c1d1()
    assert len(list(hbp_cnn.parameters())) == len(list(torch_cnn.parameters()))
    for p1, p2 in zip(hbp_cnn.parameters(), torch_cnn.parameters()):
        assert allclose(p1, p2)


def test_c1d1_forward():
    """Compare forward pass of HBP model and PyTorch c1d1."""
    set_seeds(0)
    hbp_cnn = c1d1()
    set_seeds(0)
    torch_cnn = torch_c1d1()
    x = rand(12, 1, 28, 28)
    out_torch = torch_cnn(x)
    out_hbp = hbp_cnn(x)
    assert allclose(out_torch, out_hbp)
