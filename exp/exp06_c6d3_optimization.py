"""Training of c6d3 on CIFAR-10 with SGD and CGN."""

import numpy
import torch
from torch.nn import CrossEntropyLoss, ReLU, Sigmoid, Tanh
from torch.optim import SGD
from os import path
from collections import OrderedDict
from exp.loading.load_cifar10 import CIFAR10Loader
from exp.training.first_order import FirstOrderTraining
from exp.training.second_order import SecondOrderTraining
from bpexts.optim.cg_newton import CGNewton
from bpexts.hbp.sequential import convert_torch_to_hbp
from exp.training.runner import TrainingRunner
from exp.utils import (directory_in_data, dirname_from_params, centered_list)
from exp.models.convolution import cifar10_c6d3

# directories
dirname = 'exp06_c6d3_optimization'
data_dir = directory_in_data(dirname)

# global hyperparameters
epochs = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logs_per_epoch = 5
test_batch = 100

# mapping from strings to activation functions
activation_dict = {'relu': ReLU, 'sigmoid': Sigmoid, 'tanh': Tanh}


def cifar10_sgd_train_fn(batch, lr, momentum, activation):
    """Create training instance for CIFAR-10 SGD optimization.

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
        model = cifar10_c6d3(conv_activation=act, dense_activation=act)
        loss_function = CrossEntropyLoss()
        data_loader = CIFAR10Loader(
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
            input_shape=(3, 32, 32))
        return train

    return training_fn


def sgd_grid_search():
    """Define the grid search over the hyperparameters of SGD."""
    activations = ['sigmoid', 'relu', 'tanh']
    batch_sizes = centered_list([int(2**x) for x in numpy.arange(4, 8)])
    lrs = centered_list(numpy.logspace(-4, 0, 5))
    momenta = centered_list(numpy.linspace(0, 0.9, 4))
    return [
        cifar10_sgd_train_fn(
            batch=batch, lr=lr, momentum=momentum, activation=activation)
        for batch in batch_sizes for lr in lrs for momentum in momenta
        for activation in activations
    ]


def cifar10_cgnewton_train_fn(batch, modify_2nd_order_terms, activation, lr,
                              alpha, cg_maxiter, cg_tol, cg_atol):
    """Create training instance for CIFAR10 CG experiment.


    Parameters:
    -----------
    batch : int
        Batch size
    modify_2nd_order_terms : str
        Strategy for treating 2nd-order effects of module functions:
        * `'zero'`: Yields the Generalizes Gauss Newton matrix
        * `'abs'`: BDA-PCH approximation
        * `'clip'`: Different BDA-PCH approximation
    activation : str, 'relu' or 'sigmoid' or 'tanh'
        Activation function
    lr : float
        Learning rate
    alpha : float, between 0 and 1
        Regularization in HVP, see Chen paper for more details
    cg_maxiter : int
        Maximum number of iterations for CG
    cg_tol : float
        Relative tolerance for convergence of CG
    cg_atol : float
        Absolute tolerance for convergence of CG
    """
    # batch size for evaluating metrics on the test set
    test_batch = 100

    # logging directory
    # -----------------
    # directory of run
    run_name = dirname_from_params(
        act=activation,
        opt='cgn',
        batch=batch,
        lr=lr,
        alpha=alpha,
        maxiter=cg_maxiter,
        tol=cg_tol,
        atol=cg_atol,
        mod2nd=modify_2nd_order_terms)
    logdir = path.join(data_dir, run_name)

    # training procedure
    # ------------------
    def training_fn():
        """Training function setting up the train instance."""
        # set up training and run
        act = activation_dict[activation]
        model = cifar10_c6d3(conv_activation=act, dense_activation=act)
        model = convert_torch_to_hbp(model, use_recursive=True)
        loss_function = CrossEntropyLoss()
        data_loader = CIFAR10Loader(
            train_batch_size=batch, test_batch_size=test_batch)
        optimizer = CGNewton(
            model.parameters(),
            lr=lr,
            alpha=alpha,
            cg_atol=cg_atol,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter)
        # initialize training
        train = SecondOrderTraining(
            model,
            loss_function,
            optimizer,
            data_loader,
            logdir,
            epochs,
            modify_2nd_order_terms,
            logs_per_epoch=logs_per_epoch,
            device=device,
            input_shape=(3, 32, 32))
        return train

    return training_fn


def cgn_grid_search():
    """Define the grid search over the hyperparameters of SGD."""
    batch_sizes = centered_list([int(2**x) for x in numpy.arange(4, 8)])
    mod2nds = ['abs']
    activations = ['sigmoid', 'relu', 'tanh']
    lrs = centered_list(numpy.logspace(-3, -1, 3))
    alphas = centered_list([0.01, 0.02, 0.05, 0.1])
    cg_atol = 0.
    cg_maxiter = 50
    cg_tols = centered_list([1e-5, 1e-1])
    return [
        cifar10_cgnewton_train_fn(
            batch=batch,
            modify_2nd_order_terms=mod2nd,
            activation=activation,
            lr=lr,
            alpha=alpha,
            cg_maxiter=cg_maxiter,
            cg_tol=cg_tol,
            cg_atol=cg_atol) for batch in batch_sizes for mod2nd in mod2nds
        for activation in activations for lr in lrs for alpha in alphas
        for cg_tol in cg_tols
    ]


def main(run_experiments=True):
    """Execute the experiments, return filenames of the merged runs."""
    seeds = range(1)
    labels = ['SGD']
    experiments = [
        # 1) SGD grid search
        *sgd_grid_search(),
        # 2) CGN grid search
        *cgn_grid_search()
    ]

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
