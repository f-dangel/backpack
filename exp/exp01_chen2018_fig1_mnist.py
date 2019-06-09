"""Experiments performed in Chen et al.: BDA-PCH, figure 1.

Link to the reference:
    https://arxiv.org/pdf/1802.06502v2.pdf
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from os import path
from collections import OrderedDict
from exp.models.chen2018 import mnist_model, hbp_mnist_model
from exp.loading.load_mnist import MNISTLoader
from exp.training.first_order import FirstOrderTraining
from exp.training.second_order import SecondOrderTraining
from exp.training.runner import TrainingRunner
from exp.utils import (directory_in_data, dirname_from_params)
from bpexts.optim.cg_newton import CGNewton

# global hyperparameters
batch = 500
epochs = 20
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dirname = 'exp01_reproduce_chen_figures/mnist'
data_dir = directory_in_data(dirname)
logs_per_epoch = 5


def mnist_sgd_train_fn():
    """Create training instance for MNIST SGD experiment."""
    # hyper parameters
    # ----------------
    lr = 0.1
    momentum = 0.9

    # logging directory
    # -----------------
    # directory of run
    run_name = dirname_from_params(opt='sgd', batch=batch, lr=lr, mom=momentum)
    logdir = path.join(data_dir, run_name)

    # training procedure
    # ------------------
    def training_fn():
        """Training function setting up the train instance."""
        model = mnist_model()
        loss_function = CrossEntropyLoss()
        data_loader = MNISTLoader(
            train_batch_size=batch, test_batch_size=batch)
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
            device=device)
        return train

    return training_fn


def mnist_cgnewton_train_fn(modify_2nd_order_terms):
    """Create training instance for MNIST CG experiment.

    Parameters:
    -----------
    modify_2nd_order_terms : (str)
        Strategy for treating 2nd-order effects of module functions:
        * `'zero'`: Yields the Generalized Gauss-Newton matrix
        * `'abs'`: BDA-PCH approximation
        * `'clip'`: Different BDA-PCH approximation
    """
    # hyper parameters
    # ----------------
    lr = 0.1
    alpha = 0.02
    cg_maxiter = 50
    cg_tol = 0.1
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
        model = hbp_mnist_model(
            average_input_jacobian=True, average_parameter_jacobian=True)
        loss_function = CrossEntropyLoss()
        data_loader = MNISTLoader(
            train_batch_size=batch, test_batch_size=batch)
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
            device=device)
        return train

    return train_fn


def main(run_experiments=True):
    """Execute the experiments, return filenames of the merged runs."""
    seeds = range(10)
    labels = [
        'SGD',
        'CG (GGN)',
        'CG (PCH, abs)',
        'CG (PCH, clip)',
    ]
    experiments = [
        # 1) SGD curve
        mnist_sgd_train_fn(),
        # 2) Generalized Gauss-Newton curve
        mnist_cgnewton_train_fn('zero'),
        # 3) BDA-PCH curve
        mnist_cgnewton_train_fn('abs'),
        # 4) alternative BDA-PCH curve
        mnist_cgnewton_train_fn('clip'),
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
