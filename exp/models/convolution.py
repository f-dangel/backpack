"""CNNs for testing/experiments."""

import torch.nn as nn
from numpy import prod
from bpexts.hbp.linear import HBPLinear
from bpexts.hbp.flatten import HBPFlatten
from bpexts.hbp.conv2d import HBPConv2d
from bpexts.hbp.combined_sigmoid import HBPSigmoidLinear
from bpexts.hbp.sequential import HBPSequential


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
