"""Abstract class for loading training/test sets of datasets."""

from abc import ABC, abstractmethod


class DatasetLoader(ABC):
    """Base class implementing interface for dataset loading."""
    def __init__(self,
                 train_batch_size=None,
                 train_seed=0,
                 test_batch_size=None):
        """
        Parameters:
        -----------
        train_batch_size : (int)
            Number of samples per batch in the training set, load
            entire set if left `None`
        train_seed : (int)
            Random seed for shuffling the training samples.
        test_batch_size : (int)
            Number of samples per batch in the test set, load
            entire set if left `None`
        """
        self.train_batch_size = train_batch_size
        self.train_seed = train_seed
        self.test_batch_size = test_batch_size

    @abstractmethod
    def train_loader(self):
        """Data loader of the training set.

        Returns:
        --------
        (torch.utils.data.DataLoader)
            DataLoader providing shuffled batches of the training set
        """
        pass

    @abstractmethod
    def test_loader(self):
        """Data loader of the test set.

        Returns:
        --------
        (torch.utils.data.DataLoader)
            DataLoader providing unshuffled batches of the test set
        """
        pass
