"""Test CIFAR-10 data loader."""

from bpexts.utils import torch_allclose
from .load_cifar10 import CIFAR10Loader


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


def test_test_set_size():
    """The test set size should be 10k."""
    cifar10 = CIFAR10Loader(10, 0)
    assert cifar10.test_set_size == 10000


def test_train_set_size():
    """The training set size should be 50k."""
    cifar10 = CIFAR10Loader(10, 0)
    assert cifar10.train_set_size == 50000
