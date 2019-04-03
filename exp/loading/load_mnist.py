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

from torchvision import transforms, datasets
from .load_dataset import DatasetLoader
from bpexts.utils import set_seeds
from ..utils import directory_in_data


class MNISTLoader(DatasetLoader):
    """Loading of training/test sets of MNIST."""

    # directory to store MNIST (28x28): ../dat/MNIST
    root = directory_in_data('MNIST')

    # transformation of the data
    trans = transforms.Compose([  # convert to tensor
        transforms.ToTensor(),
        # normalize
        transforms.Normalize(
            # taken from ref
            (
                0.13066048, ),
            # taken from ref
            (
                0.30810781, ))
    ])

    # download MNIST if non-existent
    train_set = datasets.MNIST(
        root=root, train=True, transform=trans, download=True)
    test_set = datasets.MNIST(
        root=root, train=False, transform=trans, download=True)


class MNISTDownsampledLoader(DatasetLoader):
    """Loading of training/test sets of MNIST downsampled to 16x16."""

    # directory to store MNIST (16x16): ../dat/MNIST_downsampled
    root = directory_in_data('MNIST_downsampled')

    # transformation of the data
    trans = transforms.Compose([
        # downsampling
        transforms.Resize((16, 16)),
        # convert to tensor
        transforms.ToTensor(),
        # normalize
        transforms.Normalize(
            # taken from ref
            (
                0.13066048, ),
            # taken from ref
            (
                0.30810781, )),
    ])

    # download MNIST if non-existent
    train_set = datasets.MNIST(
        root=root, train=True, transform=trans, download=True)
    test_set = datasets.MNIST(
        root=root, train=False, transform=trans, download=True)
