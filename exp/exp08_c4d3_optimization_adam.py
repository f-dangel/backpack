"""Training of c4d3 on CIFAR-10 with Adam."""

import numpy
import torch
from torch.nn import CrossEntropyLoss, ReLU, Sigmoid, Tanh
from torch.optim import Adam
from os import path
from collections import OrderedDict
from exp.loading.load_cifar10 import CIFAR10Loader
from exp.training.first_order import FirstOrderTraining
from bpexts.optim.cg_newton import CGNewton
from exp.training.runner import TrainingRunner
from exp.utils import (directory_in_data, dirname_from_params, centered_list)
from exp.models.convolution import cifar10_c4d3

# directories
dirname = 'exp08_c4d3_optimization/adam'
data_dir = directory_in_data(dirname)

# global hyperparameters
epochs = 30
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logs_per_epoch = 4
test_batch = 100

# mapping from strings to activation functions
activation_dict = {'relu': ReLU, 'sigmoid': Sigmoid, 'tanh': Tanh}


def cifar10_adam_train_fn(batch, lr, betas, activation):
    """Create training instance for CIFAR-10 Adam optimization.

    Parameters:
    -----------
    lr : float
        Learning rate for Adam  
    betas : (float, float) 
        Coefficients for computing running averages in Adam
    activation : str, 'relu' or 'sigmoid' or 'tanh'
        Activation function
    """
    # logging directory
    # -----------------
    # directory of run
    run_name = dirname_from_params(
        act=activation,
        opt='adam',
        batch=batch,
        lr=lr,
        b1=betas[0],
        b2=betas[1])
    logdir = path.join(data_dir, run_name)

    # training procedure
    # ------------------
    def training_fn():
        """Training function setting up the train instance."""
        act = activation_dict[activation]
        model = cifar10_c4d3(conv_activation=act, dense_activation=act)
        loss_function = CrossEntropyLoss()
        data_loader = CIFAR10Loader(
            train_batch_size=batch, test_batch_size=test_batch)
        optimizer = Adam(model.parameters(), lr=lr, betas=betas)
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
            input_shape=(3, 32, 32))
        return train

    return training_fn


def adam_grid_search():
    """Define the grid search over the hyperparameters of Adam."""
    activations = ['sigmoid']
    batch_sizes = [100, 200, 500]
    lrs = numpy.logspace(-4, 1, 6)
    betas = [(0.9, 0.999)]
    return [
        cifar10_adam_train_fn(
            batch=batch, lr=lr, betas=beta_pair, activation=activation)
        for batch in batch_sizes for lr in lrs for beta_pair in betas
        for activation in activations
    ]


def main(run_experiments=True):
    """Execute the experiments, return filenames of the merged runs."""
    seeds = range(1)
    labels = ['Adam']
    experiments = adam_grid_search()

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
