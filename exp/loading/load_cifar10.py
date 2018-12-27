"""Download CIFAR-10 dataset, provide train_loader and test_loader.

MNIST contains 60.000 rgb images of resolution 3x28x28.
The dataset is divided into a training set of size 50.000 and a test set
of size 10.000.

0) Download CIFAR-10
1) Processing pipeline:
    i) Convert to torch.Tensor
    ii) Normalize data (see reference below)

Normalization values taken from
    https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from .load_dataset import DatasetLoader
from bpexts.utils import set_seeds
from ..utils import directory_in_data


class CIFAR10Loader(DatasetLoader):
    """Loading of training/test sets of MNIST."""

    # directory to store CIFAR-10 (3x28x28): ../../dat/CIFAR10_dataset
    root = directory_in_data('CIFAR10')

    # transformation of the data
    trans = transforms.Compose(
            [  # convert to tensor
               transforms.ToTensor(),
               # normalize
               transforms.Normalize(
                   # taken from ref above
                   (0.49139968, 0.48215841, 0.44653091),
                   # taken from ref above
                   (0.24703223, 0.24348513, 0.26158784))
            ])

    # download CIFAR10 if non-existent
    train_set = datasets.CIFAR10(root=root,
                                 train=True,
                                 transform=trans,
                                 download=True)
    test_set = datasets.CIFAR10(root=root,
                                train=False,
                                transform=trans,
                                download=True)

    # size of the test set
    test_set_size = len(test_set)
    # size of the training set
    train_set_size = len(train_set)

    def train_loader(self):
        """Return loader for CIFAR-10 training data batches."""
        set_seeds(self.train_seed)
        batch = len(self.train_set)\
            if self.train_batch_size is None\
            else self.train_batch_size
        return DataLoader(dataset=self.train_set,
                          batch_size=batch,
                          shuffle=True)

    def test_loader(self):
        """Return loader for CIFAR-10 test data."""
        batch = len(self.test_set)\
            if self.test_batch_size is None\
            else self.test_batch_size
        return DataLoader(dataset=self.test_set,
                          batch_size=batch,
                          shuffle=False)
