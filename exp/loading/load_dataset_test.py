"""Test abstract data loader class."""

from .load_dataset import DatasetLoader


def test_abstract_test_size():
    """Check if attribute `test_set_size` is abstract."""
    try:
        DatasetLoader.test_set_size
    except NotImplementedError:
        pass


def test_abstract_train_size():
    """Check if attribute `train_set_size` is abstract."""
    try:
        DatasetLoader.train_set_size
    except NotImplementedError:
        pass
