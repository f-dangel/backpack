"""Test the fcnn architectures for the experiments."""

from torch import rand
from torch.nn import (Sequential, Sigmoid, Linear)
from .fcnn import fcnn
from bpexts.utils import torch_allclose, set_seeds
from numpy import prod


def torch_fcnn(
        # dimension of input
        input_size=(1, 28, 28),
        # hidden dimensions
        hidden_dims=[20],
        # dimension of output
        num_outputs=10):
    """Torch implementation of fcnn."""

    class fcnnTorch(Sequential):
        """PyTorch implementation of fcnn."""

        def __init__(self):
            in_features = prod(input_size)
            layers = []
            out_dims = hidden_dims + [num_outputs]
            for idx, out_dim in enumerate(out_dims):
                if idx == 0:
                    layers.append(
                        Linear(
                            in_features=in_features,
                            out_features=out_dim,
                            bias=True))
                else:
                    layers.append(Sigmoid())
                    layers.append(
                        Linear(
                            in_features=in_features,
                            out_features=out_dim,
                            bias=True))
                in_features = out_dim
            super().__init__(*layers)

        def forward(self, x):
            """Flatten batches before feeding through the network."""
            x = x.reshape(x.size()[0], -1)
            return super().forward(x)

    return fcnnTorch()


def test_compare_fcnn_parameters():
    """Compare the parameters of the HBP and PyTorch fcnn."""
    set_seeds(0)
    hbp_fc = fcnn()
    set_seeds(0)
    torch_fc = torch_fcnn()
    assert len(list(hbp_fc.parameters())) == len(list(torch_fc.parameters()))
    for p1, p2 in zip(hbp_fc.parameters(), torch_fc.parameters()):
        assert torch_allclose(p1, p2)


def test_fcnn_forward():
    """Compare forward pass of HBP model and PyTorch fcnn."""
    set_seeds(0)
    hbp_fc = fcnn()
    set_seeds(0)
    torch_fc = torch_fcnn()
    x = rand(12, 1, 28, 28)
    out_torch = torch_fc(x)
    out_hbp = hbp_fc(x)
    assert torch_allclose(out_torch, out_hbp)
