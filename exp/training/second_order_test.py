"""Test of training procedure."""

import torch
from torch.nn import CrossEntropyLoss
from warnings import warn
from bpexts.hbp.linear import HBPLinear
from bpexts.optim.cg_newton import CGNewton
from .second_order import SecondOrderTraining
from ..loading.load_mnist import MNISTLoader
from ..utils import directory_in_data


def simple_mnist_model_2nd_order(use_gpu=False):
    """Train on simple MNIST model using 2nd order optimizer."""
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    model = HBPLinear(784, 10)
    loss_function = CrossEntropyLoss()
    data_loader = MNISTLoader(1000, 1000)
    optimizer = CGNewton(model.parameters(), lr=0.1, alpha=0.1)
    num_epochs, logs_per_epoch = 1, 5
    modify_2nd_order_terms='abs'
    logdir = directory_in_data('test_training_simple_mnist_model')
    # initialize training
    train = SecondOrderTraining(model,
                                loss_function,
                                optimizer,
                                data_loader,
                                logdir,
                                num_epochs,
                                modify_2nd_order_terms,
                                logs_per_epoch=logs_per_epoch,
                                device=device)
    train.run()


def test_training_mnist_2nd_order_cpu():
    """Test training procedure on MNIST for CPU using CGNewton."""
    simple_mnist_model_2nd_order(use_gpu=False)


def test_training_mnist_2nd_order_gpu():
    """Test training procedure on MNIST for GPU using CGNewton."""
    if torch.cuda.is_available():
        simple_mnist_model_2nd_order(use_gpu=True)
    else:
        warn('Could not find CUDA device')
