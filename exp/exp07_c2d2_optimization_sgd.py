"""Training of c2d2 on MNIST with SGD and CGN."""

import numpy
import torch
from torch.nn import CrossEntropyLoss, ReLU, Sigmoid, Tanh
from torch.optim import SGD
from os import path
from collections import OrderedDict
from exp.loading.load_mnist import MNISTLoader
from exp.training.first_order import FirstOrderTraining
from exp.training.runner import TrainingRunner
from exp.utils import (directory_in_data, dirname_from_params)
from exp.models.convolution import mnist_c2d2

# directories
dirname = 'exp07_c2d2_optimization/sgd'
data_dir = directory_in_data(dirname)

# global hyperparameters
epochs = 5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logs_per_epoch = 10
test_batch = 500

# mapping from strings to activation functions
activation_dict = {'relu': ReLU, 'sigmoid': Sigmoid, 'tanh': Tanh}


def mnist_sgd_train_fn(batch, lr, momentum, activation):
    """Create training instance for MNIST SGD optimization.

    Parameters:
    -----------
    lr : float
        Learning rate for SGD
    momentum : float
        Momentum for SGD
    activation : str, 'relu' or 'sigmoid' or 'tanh'
        Activation function
    """
    # logging directory
    # -----------------
    # directory of run
    run_name = dirname_from_params(
        act=activation, opt='sgd', batch=batch, lr=lr, mom=momentum)
    logdir = path.join(data_dir, run_name)

    # training procedure
    # ------------------
    def training_fn():
        """Training function setting up the train instance."""
        act = activation_dict[activation]
        model = mnist_c2d2(conv_activation=act, dense_activation=act)
        loss_function = CrossEntropyLoss()
        data_loader = MNISTLoader(
            train_batch_size=batch, test_batch_size=test_batch)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
        # initialize training
        train = FirstOrderTraining(
            model,
            loss_function,
            optimizer,
            data_loader,
            logdir,
            epochs,
            logs_per_epoch=logs_per_epoch,
            device=device,
            input_shape=(1, 28, 28))
        return train

    return training_fn


def sgd_grid_search():
    """Define the grid search over the hyperparameters of SGD."""
    activations = ['tanh', 'sigmoid', 'relu']
    batch_sizes = [100, 200, 500]
    lrs = numpy.logspace(-3, -1, 3)
    momenta = numpy.linspace(0, 0.9, 3)
    return [
        mnist_sgd_train_fn(
            batch=batch, lr=lr, momentum=momentum, activation=activation)
        for batch in batch_sizes for lr in lrs for momentum in momenta
        for activation in activations
    ]


def main(run_experiments=True):
    """Execute the experiments, return filenames of the merged runs."""
    seeds = range(1)
    labels = ['SGD']
    experiments = sgd_grid_search()

    def run():
        """Run the experiments."""
        for train_fn in experiments:
            runner = TrainingRunner(train_fn)
            runner.run(seeds)

    def result_files():
        """Merge runs and return files of the merged data."""
        filenames = OrderedDict()
        for label, train_fn in zip(labels, experiments):
            runner = TrainingRunner(train_fn)
            m_to_f = runner.merge_runs(seeds)
            filenames[label] = m_to_f
        return filenames

    if run_experiments:
        run()
    return result_files()


def filenames():
    """Return filenames of the merged data.

    Returns:
    --------
    (dict)
        A dictionary with keys given by the label of the experiments.
        The associated value itself is another dictionary with keys, values
        corresponding to the metric and filename of the metric data
        respectively.
    """
    try:
        return main(run_experiments=False)
    except Exception as e:
        print("An error occured. Maybe try to re-run the experiments.")
        raise e


if __name__ == '__main__':
    main()
