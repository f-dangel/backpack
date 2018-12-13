"""Test MNIST data loader."""

import enable_import_bpexts
from bpexts.utils import torch_allclose
from load_mnist import MNISTLoader


def test_deterministic_loading():
    """Test deterministc loading of samples."""
    mnist1 = MNISTLoader(10, 0)
    train1 = mnist1.train_loader()
    train1_samples, train1_labels = next(iter(train1))

    mnist2 = MNISTLoader(10, 0)
    train2 = mnist2.train_loader()
    train2_samples, train2_labels = next(iter(train2))

    assert torch_allclose(train1_samples, train2_samples)
    assert torch_allclose(train1_labels, train2_labels)
