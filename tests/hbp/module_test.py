"""Test HBP decoration of torch.nn.Module subclasses."""

from bpexts.module import hbp_decorate
from torch.nn import Linear


def test_hbp_decorate():
    """Test decoration of a linear layer for HBP."""
    modifiedLinear = hbp_decorate(Linear)
    inputs, outputs = 5, 2
    hbp_linear = modifiedLinear(in_features=inputs, out_features=outputs)
    assert (hbp_linear.__class__.__name__) == 'HBPLinear'
