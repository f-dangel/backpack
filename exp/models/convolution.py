"""CNNs for testing/experiments."""

import torch.nn as nn
from numpy import prod
from bpexts.hbp.linear import HBPLinear
from bpexts.hbp.flatten import HBPFlatten
from bpexts.hbp.conv2d import HBPConv2d
from bpexts.hbp.combined_sigmoid import HBPSigmoidLinear
from bpexts.hbp.sequential import HBPSequential
from bpexts.utils import Flatten


def c1d1(
        # dimension of input and output of the net
        input_size=(1, 28, 28),
        num_outputs=10,
        # convolution parameters
        conv_channels=8,
        kernel_size=4,
        padding=0,
        stride=1):
    """Simple CNN with one convolution and one linear layer.

    The architecture is as follows:

      - Convolution (aka c1)
      - Sigmoid activation
      - Linear layer (fully-connected, aka d1)

    Parameters:
    -----------
    input_size : tuple(int)
        Shape of the input without batch dimension
    num_outputs : int
        Number of outputs (classes)
    conv_channels : int
        Number of feature channels created by the convolution layer

    The remaining arguments define the properties of the convolution layer.
    """
    # determine output features from convolution
    output_size = HBPConv2d.output_shape(
        input_size=(1, ) + input_size,
        out_channels=conv_channels,
        kernel_size=kernel_size,
        stride=stride)
    output_numel = prod(output_size)
    # create the model
    model = HBPSequential(
        HBPConv2d(
            in_channels=input_size[0],
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride),
        # need to flatten the image-shaped outputs of conv into vectors
        HBPFlatten(),
        HBPSigmoidLinear(
            in_features=output_numel, out_features=num_outputs, bias=True))
    return model


def mnist_c2d2(conv_activation=nn.ReLU, dense_activation=nn.ReLU):
    """CNN for MNIST with 2 convolutional and 2 fc layers.

    Modified from:
    https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a
    (remove Dropout)
    """
    return nn.Sequential(
        # Conv Layer block 1
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0),
        conv_activation(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
        conv_activation(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Flatten
        Flatten(),

        # Dense layers
        nn.Linear(9216, 128),
        dense_activation(),
        nn.Linear(128, 10))


def cifar10_c6d3(conv_activation=nn.ReLU, dense_activation=nn.ReLU):
    """CNN for CIFAR-10 dataset with 6 convolutional and 3 fc layers.

    Modified from:
    https://github.com/Zhenye-Na/deep-learning-uiuc/tree/master/assignments/mp3
    (remove Dropout, Dropout2d and BatchNorm2d)
    """
    return nn.Sequential(
        # Conv Layer block 1
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
        conv_activation(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        conv_activation(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Conv Layer block 2
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        conv_activation(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        conv_activation(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Conv Layer block 3
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        conv_activation(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        conv_activation(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Flatten
        Flatten(),

        # Dense layers
        nn.Linear(4096, 1024),
        dense_activation(),
        nn.Linear(1024, 512),
        dense_activation(),
        nn.Linear(512, 10))
