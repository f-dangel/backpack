"""FCNNs for testing/experiments."""

import torch.nn as nn
from numpy import prod
from bpexts.hbp.linear import HBPLinear
from bpexts.hbp.view import HBPViewBatchFlat
from bpexts.hbp.combined_sigmoid import HBPSigmoidLinear
from bpexts.hbp.sequential import HBPSequential


def fcnn(
        # dimension of input
        input_size=(1, 28, 28),
        # hidden dimensions
        hidden_dims=[20],
        # dimension of output
        num_outputs=10):
    """FCNN with linear layers (all with Sigmoid activations except last).

    The architecture is as follows:

      - Linear, Sigmoid, Linear, Sigmoid, ..., Linear

    Parameters:
    -----------
    input_size : tuple(int)
        Shape of the input without batch dimension
    hidden_dims : tuple(int)
        Number of neurons in hidden layers
    num_outputs : int
        Number of outputs (classes)
    """
    in_features = prod(input_size)
    layers = [HBPViewBatchFlat()]
    out_dims = hidden_dims + [num_outputs]
    for idx, out_dim in enumerate(out_dims):
        if idx == 0:
            layers.append(
                HBPLinear(
                    in_features=in_features, out_features=out_dim, bias=True))
        else:
            layers.append(
                HBPSigmoidLinear(
                    in_features=in_features, out_features=out_dim, bias=True))

        in_features = out_dim
    model = HBPSequential(*layers)
    return model
