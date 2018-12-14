"""Test of training procedure."""

import torch
from os import path
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import SGD
from warnings import warn
from training import FirstOrderTraining
from load_mnist import MNISTLoader


# directory to log quantities : ../dat
parent_dir = path.dirname(
        path.dirname(path.realpath(__file__)))


def simple_mnist_model(use_gpu=False):
    """Train on simple MNIST model."""
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    model = Linear(784, 10)
    loss_function = CrossEntropyLoss()
    data_loader = MNISTLoader(10000)
    optimizer = SGD(model.parameters(), lr=0.1)
    data_dir = 'dat/test_training_simple_mnist_model'
    logdir = path.join(parent_dir, data_dir)
    # initialize training
    train = FirstOrderTraining(model, loss_function,
                               optimizer, data_loader, logdir)
    train.run(num_epochs=1, device=device)


def test_training_mnist_cpu():
    """Test training procedure on MNIST for CPU."""
    simple_mnist_model(use_gpu=False)


def test_training_mnist_gpu():
    """Try to run training procedure on MNIST for GPU."""
    if torch.cuda.is_available():
        simple_mnist_model(use_gpu=True)
    else:
        warn('Could not find CUDA device')
