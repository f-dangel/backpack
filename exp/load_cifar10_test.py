"""Test CIFAR-10 data loader."""

from bpexts.utils import torch_allclose
from load_cifar10 import CIFAR10Loader


def test_deterministic_loading():
    """Test deterministc loading of samples."""
    cifar10_1 = CIFAR10Loader(10, 0)
    train1 = cifar10_1.train_loader()
    train1_samples, train1_labels = next(iter(train1))

    cifar10_2 = CIFAR10Loader(10, 0)
    train2 = cifar10_2.train_loader()
    train2_samples, train2_labels = next(iter(train2))

    assert torch_allclose(train1_samples, train2_samples)
    assert torch_allclose(train1_labels, train2_labels)
