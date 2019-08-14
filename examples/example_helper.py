"""
Helper functions for BackPACK Examples
"""

from typing import Callable
from backpack.core.layers import Flatten
from torch.nn import CrossEntropyLoss, Sequential, Conv2d, ReLU, Linear, AvgPool2d, MaxPool2d
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def get_model(model_name):
    available_models = ["2-conv-1-linear"]
    if model_name not in available_models:
        raise NotImplementedError(
            "Model {} unknown ".format(model_name) +
            "Available models: {}".format(available_models)
        )

    if model_name == "2-conv-1-linear":
        return Sequential(
            Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
            ),
            ReLU(),
            MaxPool2d(2, stride=2),
            Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
            ),
            ReLU(),
            MaxPool2d(2, stride=2),
            Flatten(),
            Linear(512, 10),
        )


def mnist_loader(train=True, batch_size=128):
    """
    Returns a DataLoader for MNIST

    :param train: Return the training set if True, the test set if False
    :param batch_size: Batch size used in the loader
    :return:
    """
    return DataLoader(
        datasets.MNIST(
            './data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size,
        shuffle=True
    )


def mnist_train_loader(batch_size=128):
    """
    Dataloader for the MNIST Training set using `batch_size`
    """
    return mnist_loader(train=True, batch_size=batch_size)


def mnist_test_loader(batch_size=128):
    """
    Dataloader for the MNIST Test set using `batch_size`
    """
    return mnist_loader(train=False, batch_size=batch_size)


def accuracy(model, loader):
    """
    Compute the accuracy of the `model` on the data in the `loader`
    """
    acc = 0
    N = 0
    for batch_idx, (data, target) in enumerate(loader):
        pred = model(data).argmax(dim=1, keepdim=True)
        acc += pred.eq(target.view_as(pred)).float().sum().item()
        N += data.shape[0]
    acc /= N
    return acc


def loss(model, loader):
    """
    Compute the cross entropy loss of the `model` on the data in the `loader`
    """
    l = 0
    N = 0
    for batch_idx, (data, target) in enumerate(loader):
        l += CrossEntropyLoss(reduction="sum")(model(data), target).item()
        N += data.shape[0]
    l /= N
    return l


def iterate(loader, epochs, iter: Callable, use_tqdm=False):
    """
    Iterates over the `train_loader` for `epochs`, calling the function `iter`
    at each iteration.

    The signature of `iter`: `iter(epoch, batch_idx, data, target) -> None`

    :param loader: data loader
    :param epochs: Number of epochs to run
    :param iter: iteration function
    """
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in tqdm(enumerate(loader), leave=False, total=len(loader)):
            iter(epoch, batch_idx, data, target)


def rgb_to_u(xs):
    return list([x / 255.0 for x in xs])


COLORS = {
    "yellow": rgb_to_u([221, 170, 51]),
    "red": rgb_to_u([187, 85, 102]),
    "blue": rgb_to_u([0, 68, 136]),
    "black": rgb_to_u([0, 0, 0]),
}


def example_1_plot(losses, accuracies, norms, variances):
    fig = plt.figure(figsize=(12, 6))

    LINEWIDTH = 3

    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.plot(losses, linewidth=LINEWIDTH, color=COLORS["blue"])
    ax2.plot(accuracies, linewidth=LINEWIDTH, color=COLORS["red"])
    ax3.plot(norms, linewidth=LINEWIDTH, color=COLORS["black"])
    ax4.plot(variances, linewidth=LINEWIDTH, color=COLORS["yellow"])

    ax1.set_title("Loss")
    ax2.set_title("Accuracy")
    ax3.set_title("Gradient norm")
    ax4.set_title("Gradient variance")

    ax1.set_ylabel("Cross-entropy")
    ax2.set_ylabel("Accuracy")
    ax3.set_ylabel("Gradient norm")
    ax4.set_ylabel("Gradient variance")

    ax3.set_xlabel("Iteration")
    ax4.set_xlabel("Iteration")

    ax1.set_ylim([0, ax1.get_ylim()[1]])
    ax2.set_ylim([0, 1])
    ax3.set_ylim([0, ax3.get_ylim()[1]])
    ax4.set_ylim([0, ax4.get_ylim()[1]])

    ax1.grid(axis="x")
    ax2.grid(axis="x")
    ax3.grid(axis="x")
    ax4.grid(axis="x")

    plt.show()


def example_2_plot(losses, accuracies):
    fig = plt.figure(figsize=(12, 4))

    LINEWIDTH = 3

    gs = GridSpec(1, 2, figure=fig, hspace=0.25, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.plot(losses, linewidth=LINEWIDTH, color=COLORS["blue"])
    ax2.plot(accuracies, linewidth=LINEWIDTH, color=COLORS["red"])

    ax1.set_title("Loss")
    ax2.set_title("Accuracy")

    ax1.set_ylabel("Cross-entropy")
    ax2.set_ylabel("Accuracy")

    ax1.set_xlabel("Iteration")
    ax1.set_xlabel("Iteration")

    ax1.set_ylim([0, ax1.get_ylim()[1]])
    ax2.set_ylim([0, 1])

    ax1.grid(axis="x")
    ax2.grid(axis="x")

    plt.show()


def log(epoch, EPOCHS, batch_idx, train_loader, batch_loss, batch_accuracy, additional=""):
    print(
        "Epoch {}/{}".format(epoch, EPOCHS),
        "Batch {}/{}".format(batch_idx, len(train_loader)),
        "Batch loss: {0:.2f}".format(batch_loss.item()),
        "Batch accuracy: {0:.2f}".format(batch_accuracy),
        additional
    )
