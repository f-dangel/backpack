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

from os import path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
# enable import of bpexts in parent directory
import enable_import_bpexts
from bpexts.utils import set_seeds

# directory to store CIFAR-10 (3x28x28): ../dat/CIFAR10_dataset
parent_dir = path.dirname(
        path.dirname(path.realpath(__file__)))
data_dir = 'dat/CIFAR10'
root = path.join(parent_dir, data_dir)

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


def train_loader(batch_size=None,
                 seed=0):
    """Return loader for CIFAR-10 training data batches.

    Use entire train set if batch_size is unspecified.
    """
    set_seeds(seed)
    batch_size = len(train_set) if batch_size is None else batch_size
    return DataLoader(dataset=train_set,
                      batch_size=batch_size,
                      shuffle=True)


def test_loader(batch_size=None):
    """Return loader for CIFAR-10 test data.

    Use entire test set if batch_size is unspecified.
    """
    batch_size = len(test_set) if batch_size is None else batch_size
    return DataLoader(dataset=test_set,
                      batch_size=batch_size,
                      shuffle=False)
