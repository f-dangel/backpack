import torch
import torchvision


def download_mnist():
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


def load_mnist_data(batch_size=64, shuffle=True):
    """Return (inputs, labels) for an MNIST mini-batch."""
    mnist_dataset = download_mnist()
    mnist_loader = torch.utils.data.dataloader.DataLoader(
        mnist_dataset, batch_size=batch_size, shuffle=shuffle,
    )

    X, y = next(iter(mnist_loader))
    return X, y
