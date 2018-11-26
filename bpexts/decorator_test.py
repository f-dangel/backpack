"""Test decorator of torch.nn.Module subclasses."""

import torch.nn
from .decorator import decorate


def test_decorated_linear_properties():
    """Test name and docstring of decorated torch.nn.Linear."""
    DecoratedAffine = decorate(torch.nn.Linear)
    assert DecoratedAffine.__name__ == 'DecoratedLinear'
    assert DecoratedAffine.__doc__.startswith('[Decorated by bpexts]')
