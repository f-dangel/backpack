"""Test of first-order training procedure."""

import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import SGD
from warnings import warn
from .first_order import FirstOrderTraining
from ..loading.load_mnist import MNISTLoader
from ..utils import directory_in_data


def simple_mnist_model_1st_order(use_gpu=False):
    """Train on simple MNIST model, using SGD."""
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    model = Linear(784, 10)
    loss_function = CrossEntropyLoss()
    data_loader = MNISTLoader(1000)
    optimizer = SGD(model.parameters(), lr=0.1)
    num_epochs, logs_per_epoch = 1, 15
    logdir = directory_in_data('test_training_simple_mnist_model')
    # initialize training
    train = FirstOrderTraining(model,
                               loss_function,
                               optimizer,
                               data_loader,
                               logdir,
                               num_epochs,
                               logs_per_epoch=logs_per_epoch,
                               device=device)
    train.run()


def test_training_mnist_1st_order_cpu():
    """Test training procedure on MNIST for CPU using SGD."""
    simple_mnist_model_1st_order(use_gpu=False)


def test_training_mnist_1st_order_gpu():
    """Try to run training procedure on MNIST for GPU using SGD."""
    if torch.cuda.is_available():
        simple_mnist_model_1st_order(use_gpu=True)
    else:
        warn('Could not find CUDA device')
