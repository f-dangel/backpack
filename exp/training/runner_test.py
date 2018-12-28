"""Test runner of training for different seeds."""

import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import SGD
from warnings import warn
from .runner import TrainingRunner
from .first_order import FirstOrderTraining
from ..loading.load_mnist import MNISTLoader
from ..utils import directory_in_data


def training_fn_on_device(use_gpu):
    """Return function that creates the training instance."""
    def training_fn():
        """Return training instance."""
        device = torch.device('cuda:0' if use_gpu else 'cpu')
        model = Linear(784, 10)
        loss_function = CrossEntropyLoss()
        data_loader = MNISTLoader(1000, 1000)
        optimizer = SGD(model.parameters(), lr=0.1)
        num_epochs, logs_per_epoch = 1, 5
        logdir = directory_in_data('test_training_simple_mnist_model_{}'
                                   .format('gpu' if use_gpu else 'cpu'))
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
    return training_fn
 

def simple_mnist_model_1st_order(use_gpu=False):
    """Train on simple MNIST model, using SGD, for multiple seeds."""
    training_fn = training_fn_on_device(use_gpu)
    runner = TrainingRunner(training_fn)
    seeds = [1,2,3]
    runner.run(seeds)
    runner.merge_runs(seeds)


def test_runner_mnist_1st_order_cpu():
    """Train on MNIST for CPU using SGD, for multiple seeds."""
    simple_mnist_model_1st_order(use_gpu=False)


def test_runner_mnist_1st_order_gpu():
    """Try to run training on MNIST for GPU using SGD, multiple seeds."""
    if torch.cuda.is_available():
        simple_mnist_model_1st_order(use_gpu=True)
    else:
        warn('Could not find CUDA device')
