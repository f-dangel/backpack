"""Test of model architectures from Chen et al.: BDA-PCH (2018)."""

import enable_import_bpexts
from bpexts.utils import set_seeds, torch_allclose
from models_chen2018 import (original_mnist_model,
                             separated_mnist_model,
                             original_cifar10_model,
                             separated_cifar10_model)

from torch import randn


def test_forward_mnist_models():
    """Check same behaviour of original and separated MNIST model."""
    input = randn(2, 784)
    original = original_mnist_model(seed=13)
    separated = separated_mnist_model(seed=13)
    assert torch_allclose(original(input), separated(input), atol=1E-5)


def test_forward_cifar10_models():
    """Check same behaviour of original and separated CIFAR-10 model."""
    input = randn(2, 3072)
    original = original_cifar10_model(seed=11)
    separated = separated_cifar10_model(seed=11)
    assert torch_allclose(original(input), separated(input), atol=1E-5)
