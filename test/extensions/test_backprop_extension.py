"""Test custom extensions for backprop_extension."""

import pytest
from torch.nn import Linear, Module

from backpack.extensions import BatchGrad, Variance
from backpack.extensions.firstorder.base import FirstOrderModuleExtension


def test_set_custom_extension():
    """Test the method set_custom_extension of BackpropExtension."""

    class _A(Module):
        pass

    class _ABatchGrad(FirstOrderModuleExtension):
        pass

    class _AVariance(FirstOrderModuleExtension):
        pass

    class _MyLinearBatchGrad(FirstOrderModuleExtension):
        pass

    grad_batch = BatchGrad()

    # Set module extension
    grad_batch.set_module_extension(_A, _ABatchGrad())

    # setting again should raise a ValueError
    with pytest.raises(ValueError):
        grad_batch.set_module_extension(_A, _ABatchGrad())

    # setting again with overwrite
    grad_batch.set_module_extension(_A, _ABatchGrad(), overwrite=True)

    # in a different extension, set another extension for the same module
    variance = Variance()
    variance.set_module_extension(_A, _AVariance())

    # set an extension for an already existing extension
    with pytest.raises(ValueError):
        grad_batch.set_module_extension(Linear, _MyLinearBatchGrad())

    grad_batch.set_module_extension(Linear, _MyLinearBatchGrad(), overwrite=True)
