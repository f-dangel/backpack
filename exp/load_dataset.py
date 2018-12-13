"""Abstract class for loading training/test sets of datasets."""

from abc import ABC, abstractmethod


class DatasetLoader(ABC):
    """Base class implementing interface for dataset loading."""

    @abstractmethod
    def train_loader(self,
                     batch_size=None,
                     seed=0):
        """Data loader of the training set.

        Parameters:
        -----------
        batch_size : (int)
            Number of samples per batch, load entire set if left `None`
        seed : (int)
            Random seed for shuffling the training samples.

        Returns:
        --------
        (torch.utils.data.DataLoader)
            DataLoader providing shuffled batches of the training set
        """
        pass

    @abstractmethod
    def test_loader(self,
                    batch_size=None):
        """Data loader of the test set.

        Parameters:
        -----------
        batch_size : (int)
            Number of samples per batch, load entire set if left `None`

        Returns:
        --------
        (torch.utils.data.DataLoader)
            DataLoader providing unshuffled batches of the test set
        """
        pass
