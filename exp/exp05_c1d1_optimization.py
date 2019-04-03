"""Training of c1d1 from exp03 on MNIST with batch size 500."""

import numpy
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from os import path
from collections import OrderedDict
from exp.loading.load_mnist import MNISTLoader
from exp.training.first_order import FirstOrderTraining
from exp.training.second_order import SecondOrderTraining
from exp.training.runner import TrainingRunner
from exp.utils import (directory_in_data, dirname_from_params)
from bpexts.optim.cg_newton import CGNewton
from exp03_c1d1_hessian import c1d1_model

# directories
dirname = 'exp05_c1d1_optimization'
data_dir = directory_in_data(dirname)

# global hyperparameters
batch, test_batch = 500, 1000
epochs = 5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logs_per_epoch = 20


def mnist_sgd_train_fn(lr, momentum):
    """Create training instance for MNIST SGD optimization.

    Parameters:
    -----------
    lr : (float)
        Learning rate for SGD
    momentum : (float)
        Momentum for SGD
    """
    # logging directory
    # -----------------
    # directory of run
    run_name = dirname_from_params(opt='sgd', batch=batch, lr=lr, mom=momentum)
    logdir = path.join(data_dir, run_name)

    # training procedure
    # ------------------
    def training_fn():
        """Training function setting up the train instance."""
        model = c1d1_model()
        # NOTE: Important line, deactivate extension hooks/buffers!
        model.disable_exts()
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
    # commented out: whole grid search
    lrs = [0.1]
    # lrs = numpy.logspace(-4, 0, 5)
    momenta = [0.3]
    # momenta = numpy.linspace(0, 0.9, 4)
    return [
        mnist_sgd_train_fn(lr=lr, momentum=momentum) for lr in lrs
        for momentum in momenta
    ]


def mnist_cgnewton_train_fn(lr, alpha, modify_2nd_order_terms, cg_tol):
    """Create training instance for MNIST CG experiment.

    Parameters:
    -----------
    modify_2nd_order_terms : (str)
        Strategy for treating 2nd-order effects of module functions:
        * `'zero'`: Yields the Generalized Gauss-Newton matrix
        * `'abs'`: BDA-PCH approximation
        * `'clip'`: Different BDA-PCH approximation
    """
    # fixed hyper parameters
    # ----------------
    cg_maxiter = 50
    cg_atol = 0

    # logging directory
    # -----------------
    # directory of run
    run_name = dirname_from_params(
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
    def train_fn():
        """Training function setting up the train instance."""
        # set up training and run
        model = c1d1_model()
        loss_function = CrossEntropyLoss()
        data_loader = MNISTLoader(
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
            input_shape=(1, 28, 28))
        return train

    return train_fn


def cgn_grid_search():
    """Define the grid search over the hyperparameters of CGN."""
    # commented out: grid search parameters
    # need to convert to Python float for correct casting to torch
    lrs = [0.1]
    # lrs = numpy.logspace(-4, 0, 5)
    lrs = list(map(float, lrs))
    alphas = [0.1]
    # alphas = numpy.logspace(-3, -1, 3)
    alphas = list(map(float, alphas))
    cg_tols = [0.001]
    # cg_tols = numpy.logspace(-5, -1, 3)
    cg_tols = list(map(float, cg_tols))
    modify_2nd_order_terms = ["abs"]
    return [
        mnist_cgnewton_train_fn(
            lr=lr, alpha=alpha, modify_2nd_order_terms=mod2nd, cg_tol=cg_tol)
        for lr in lrs for alpha in alphas for mod2nd in modify_2nd_order_terms
        for cg_tol in cg_tols
    ]


def main(run_experiments=True):
    """Execute the experiments, return filenames of the merged runs."""
    seeds = range(10)
    labels = ['SGD', "PCH-abs"]
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
