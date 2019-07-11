from torch.nn import Sequential, Conv2d, MaxPool2d, ReLU, Linear, CrossEntropyLoss
from backpack.gradient.layers import Flatten
from backpack.gradient import extend


def make_model() -> Sequential:
    return Sequential(
        extend(Conv2d(1, 20, 5, 1)),
        extend(ReLU()),
        extend(MaxPool2d(2, 2)),
        extend(Conv2d(20, 50, 5, 1)),
        extend(ReLU()),
        extend(MaxPool2d(2, 2)),
        extend(Flatten()),
        extend(Linear(4 * 4 * 50, 500)),
        extend(ReLU()),
        extend(Linear(500, 10)),
    )


def make_quadratic_model() -> Sequential:
    return Sequential(
        extend(Flatten()),
        extend(Linear(784, 10)),
    )


def make_lossfunc():
    return extend(CrossEntropyLoss())
