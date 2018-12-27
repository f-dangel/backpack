"""Abstract class for loading training/test sets of datasets."""

from abc import ABC
from torch.utils.data import DataLoader
from bpexts.utils import set_seeds


class DatasetLoader(ABC):
    """Base class implementing interface for dataset loading."""
    def __init__(self,
                 train_batch_size=None,
                 train_seed=None,
                 test_batch_size=None):
        """
        Parameters:
        -----------
        train_batch_size : (int)
            Number of samples per batch in the training set, load
            entire set if left `None`
        train_seed : (int or None)
            Random seed for shuffling the training samples.
            No reset of the seed if left `None`
        test_batch_size : (int)
            Number of samples per batch in the test set, load
            entire set if left `None`
        """
        self.train_batch_size = train_batch_size
        self.train_seed = train_seed
        self.test_batch_size = test_batch_size

    @property
    def test_set_size(self):
        """Return the size of the test set."""
        return len(self.test_set)

    @property
    def train_set_size(self):
        """Return the size of the training set."""
        return len(self.train_set)

    @property
    def train_set(self):
        """Dataset correponding to the training set."""
        raise NotImplementedError('Please define the training set size')

    @property
    def test_set(self):
        """Dataset corresponding to the test set."""
        raise NotImplementedError('Please define the test set size')

    def train_loader(self):
        """Data loader of the training set.

        Returns:
        --------
        (torch.utils.data.DataLoader)
            DataLoader providing shuffled batches of the training set
        """
        set_seeds(self.train_seed)
        batch = len(self.train_set)\
            if self.train_batch_size is None\
            else self.train_batch_size
        return DataLoader(dataset=self.train_set,
                          batch_size=batch,
                          shuffle=True)

    def test_loader(self):
        """Data loader of the test set.

        Returns:
        --------
        (torch.utils.data.DataLoader)
            DataLoader providing unshuffled batches of the test set
        """
        batch = len(self.test_set)\
            if self.test_batch_size is None\
            else self.test_batch_size
        return DataLoader(dataset=self.test_set,
                          batch_size=batch,
                          shuffle=False)
