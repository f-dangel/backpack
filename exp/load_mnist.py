"""Download MNIST dataset, provide train_loader and test_loader.

MNIST contains 60.000 gray-scale images of resolution 28x28.
The dataset is divided into a training set of size 50.000 and a test set
of size 10.000.

0) Download MNIST
1) Processing pipeline:
    i) Convert to torch.Tensor
    ii) Normalize data (see reference below)

Normalization values taken from
    https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""

from os import path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from load_dataset import DatasetLoader
from bpexts.utils import set_seeds


class MNISTLoader(DatasetLoader):
    """Loading of training/test sets of MNIST."""

    # directory to store MNIST (28x28): ../dat/MNIST_dataset
    parent_dir = path.dirname(
            path.dirname(path.realpath(__file__)))
    data_dir = 'dat/MNIST'
    root = path.join(parent_dir, data_dir)

    # transformation of the data
    trans = transforms.Compose(
            [  # convert to tensor
               transforms.ToTensor(),
               # normalize
               transforms.Normalize(
                   # taken from ref
                   (0.13066048,),
                   # taken from ref
                   (0.30810781,))
            ])

    # download MNIST if non-existent
    train_set = datasets.MNIST(root=root,
                               train=True,
                               transform=trans,
                               download=True)
    test_set = datasets.MNIST(root=root,
                              train=False,
                              transform=trans,
                              download=True)

    def train_loader(self):
        """Return loader for MNIST training data batches."""
        set_seeds(self.train_seed)
        batch = len(self.train_set)\
            if self.train_batch_size is None\
            else self.train_batch_size
        return DataLoader(dataset=self.train_set,
                          batch_size=batch,
                          shuffle=True)

    def test_loader(self):
        """Return loader for MNIST test data."""
        batch = len(self.test_set)\
            if self.test_batch_size is None\
            else self.test_batch_size
        return DataLoader(dataset=self.test_set,
                          batch_size=batch,
                          shuffle=False)
