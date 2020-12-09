"""Utility functions for examples."""
import torch
import torchvision


def load_mnist_dataset():
    """Download and normalize MNIST training data."""
    mnist_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
        download=True,
    )
    return mnist_dataset


def get_mnist_dataloader(batch_size=64, shuffle=True):
    """Returns a dataloader for MNIST"""
    return torch.utils.data.dataloader.DataLoader(
        load_mnist_dataset(),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def load_one_batch_mnist(batch_size=64, shuffle=True):
    """Return a single batch (inputs, labels) of MNIST data."""
    dataloader = get_mnist_dataloader(batch_size, shuffle)
    X, y = next(iter(dataloader))
    return X, y
