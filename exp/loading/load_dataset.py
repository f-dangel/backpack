"""Abstract class for loading training/test sets of datasets."""

from abc import ABC
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from bpexts.utils import set_seeds


class DatasetLoader(ABC):
    """Base class implementing interface for dataset loading."""
    def __init__(self,
                 train_batch_size=None,
                 test_batch_size=None):
        """
        Parameters:
        -----------
        train_batch_size : (int)
            Number of samples per batch in the training set, load
            entire set if left `None`
        test_batch_size : (int)
            Number of samples per batch in the test set, load
            entire set if left `None`
        """
        self.train_batch_size = (self.train_set_size if train_batch_size
                                 is None else train_batch_size)
        self.test_batch_size = (self.test_set_size if test_batch_size
                                is None else test_batch_size)

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

    def train_loader(self, pin_memory=False):
        """Data loader of the training set.

        Parameters:
        -----------
        pin_memory : (bool)
            Load data to pinned memory. This speeds up host-to-device
            transfers when training on the GPU

        Returns:
        --------
        (torch.utils.data.DataLoader)
            DataLoader providing shuffled batches of the training set
        """
        return DataLoader(dataset=self.train_set,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          pin_memory=pin_memory)

    def train_loss_loader(self, pin_memory=False):
        """Load random subset of train set of same size as test set.

        Parameters:
        -----------
        pin_memory : (bool)
            Load data to pinned memory. This speeds up host-to-device
            transfers when training on the GPU


        Returns:
        --------
        (torch.utils.data.DataLoader)
            DataLoader providing a random subset of the training set
            of same size as the test set in batches of size 
            `self.train_batch_size`.
        """
        indices = np.random.choice(self.train_set_size,
                                   size=self.test_set_size,
                                   replace=False)
        sampler = SubsetRandomSampler(indices)
        return DataLoader(dataset=self.train_set,
                          batch_size=self.train_batch_size,
                          sampler=sampler,
                          pin_memory=pin_memory)

    def test_loader(self, pin_memory=False):
        """Data loader of the test set.

        Parameters:
        -----------
        pin_memory : (bool)
            Load data to pinned memory. This speeds up host-to-device
            transfers when training on the GPU


        Returns:
        --------
        (torch.utils.data.DataLoader)
            DataLoader providing unshuffled batches of the test set
        """
        return DataLoader(dataset=self.test_set,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          pin_memory=pin_memory)
