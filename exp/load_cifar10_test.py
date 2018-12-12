"""Test CIFAR-10 data loader."""

import enable_import_bpexts
from bpexts.utils import torch_allclose
from load_cifar10 import train_loader, test_loader


def test_deterministic_loading():
    """Test deterministc loading of samples."""
    train1 = train_loader(10)
    train1_samples, train1_labels = next(iter(train1))

    train2 = train_loader(10)
    train2_samples, train2_labels = next(iter(train2))

    assert torch_allclose(train1_samples, train2_samples)
    assert torch_allclose(train1_labels, train2_labels)
