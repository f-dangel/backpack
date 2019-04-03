"""Test MNIST data loader."""

from bpexts.utils import torch_allclose, set_seeds
from .load_mnist import MNISTLoader, MNISTDownsampledLoader

loaders = [MNISTLoader, MNISTDownsampledLoader]


def test_deterministic_loading():
    """Test deterministc loading of samples."""
    for loader in loaders:
        set_seeds(0)
        mnist1 = loader(10, 10)
        train1 = mnist1.train_loader()
        train1_samples, train1_labels = next(iter(train1))

        set_seeds(0)
        mnist2 = loader(10, 10)
        train2 = mnist2.train_loader()
        train2_samples, train2_labels = next(iter(train2))

        assert torch_allclose(train1_samples, train2_samples)
        assert torch_allclose(train1_labels, train2_labels)


def test_test_set_size():
    """The test set size should be 10k."""
    for loader in loaders:
        mnist = loader(10, 10)
        assert mnist.test_set_size == 10000


def test_train_set_size():
    """The training set size should be 60k."""
    for loader in loaders:
        mnist = loader(10, 10)
        assert mnist.train_set_size == 60000


def test_train_loss_loader():
    """Should return randum subset of training set."""
    for loader in loaders:
        mnist = loader(1000, 10)
        samples = 0
        for (inputs, labels) in mnist.train_loss_loader():
            samples += labels.size()[0]
        assert samples == mnist.test_set_size


def test_downsampled_size():
    """Check the resolution of the downsampled MNIST images."""
    mnist = MNISTDownsampledLoader(10, 10)
    train = mnist.train_loader()
    train_samples, train_labels = next(iter(train))
    assert tuple(train_samples.size()) == (10, 1, 16, 16)
    test = mnist.test_loader()
    test_samples, test_labels = next(iter(test))
    assert tuple(test_samples.size()) == (10, 1, 16, 16)
