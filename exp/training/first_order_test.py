"""Test of first-order training procedure."""

import numpy 
import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import SGD
from warnings import warn
from .first_order import FirstOrderTraining
from ..loading.load_mnist import MNISTLoader
from ..utils import directory_in_data
from bpexts.utils import set_seeds


logdir = directory_in_data('test_training_simple_mnist_model')
num_epochs, logs_per_epoch = 1, 15


def simple_mnist_model_1st_order(use_gpu=False):
    """Train on simple MNIST model, using SGD."""
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    model = Linear(784, 10)
    loss_function = CrossEntropyLoss()
    data_loader = MNISTLoader(1000, 1000)
    optimizer = SGD(model.parameters(), lr=0.1)
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


def training_example(seed, test_batch, use_gpu=False):
    """Training instance setting seed and test batch size in advance."""
    set_seeds(seed)
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    model = Linear(784, 10)
    loss_function = CrossEntropyLoss()
    data_loader = MNISTLoader(1000, test_batch)
    optimizer = SGD(model.parameters(), lr=0.1)
    # initialize training
    train = FirstOrderTraining(model,
                               loss_function,
                               optimizer,
                               data_loader,
                               logdir,
                               num_epochs,
                               logs_per_epoch=logs_per_epoch,
                               device=device)
    return train
 

def test_test_loss_and_accuracy(use_gpu=False):
    """Compute test loss/accuracy from different train set batch sizes."""
    train1 = training_example(0, 130, use_gpu)
    train1.run()
    loss1, acc1 = train1.test_loss_and_accuracy()

    train2 = training_example(0, 900, use_gpu)
    train2.run()
    loss2, acc2 = train2.test_loss_and_accuracy()

    train3 = training_example(0, None, use_gpu)
    train3.run()
    loss3, acc3 = train3.test_loss_and_accuracy()

    assert numpy.isclose(loss1, loss2)
    assert numpy.isclose(acc1, acc2)
    assert numpy.isclose(loss2, loss3)
    assert numpy.isclose(acc2, acc3)
