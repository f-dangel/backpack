"""
Experiments performed in Chen et al.: BDA-PCH, figure 2,
with different parameter splitting.

Link to the reference:
    https://arxiv.org/pdf/1802.06502v2.pdf
"""

import torch
from torch.nn import CrossEntropyLoss
from os import path, makedirs
import matplotlib.pyplot as plt
from .models.chen2018 import original_cifar10_model
from .loading.load_cifar10 import CIFAR10Loader
from .training.second_order import SecondOrderTraining
from .training.runner import TrainingRunner
from .plotting.plotting import OptimizationPlot
from .utils import (directory_in_data,
                    directory_in_fig,
                    dirname_from_params)
from bpexts.optim.cg_newton import CGNewton
from bpexts.hbp.parallel.sequential import HBPParallelSequential


# global hyperparameters
batch = 500
epochs = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dirname = 'exp02_chen_splitting/cifar10'
data_dir = directory_in_data(dirname)
fig_dir = directory_in_fig(dirname)
logs_per_epoch = 1


def cifar10_cgnewton_train_fn(modify_2nd_order_terms, max_blocks):
    """Create training instance for CIFAR10 CG experiment

    Trainable parameters (weights and bias) will be split into
    subgroups during optimization.


    Parameters:
    -----------
    modify_2nd_order_terms : (str)
        Strategy for treating 2nd-order effects of module functions:
        * `'zero'`: Yields the Generalizes Gauss Newton matrix
        * `'abs'`: BDA-PCH approximation
        * `'clip'`: Different BDA-PCH approximation
    max_blocks : (int)
        * Split parameters per layer into subblocks during
          optimization (less if not divisible)
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
    run_name = dirname_from_params(opt='cgn',
                                   batch=batch,
                                   lr=lr,
                                   alpha=alpha,
                                   maxiter=cg_maxiter,
                                   tol=cg_tol,
                                   atol=cg_atol,
                                   mod2nd=modify_2nd_order_terms,
                                   blocks=max_blocks)
    logdir = path.join(data_dir, run_name)

    # training procedure
    # ------------------
    def training_fn():
        """Training function setting up the train instance."""
        # set up training and run
        model = original_cifar10_model()
        # split into parallel modules
        model = HBPParallelSequential(max_blocks, *list(model.children()))
        loss_function = CrossEntropyLoss()
        data_loader = CIFAR10Loader(train_batch_size=batch,
                                    test_batch_size=batch)
        optimizer = CGNewton(model.parameters(),
                             lr=lr,
                             alpha=alpha,
                             cg_atol=cg_atol,
                             cg_tol=cg_tol,
                             cg_maxiter=cg_maxiter)
        # initialize training
        train = SecondOrderTraining(model,
                                    loss_function,
                                    optimizer,
                                    data_loader,
                                    logdir,
                                    epochs,
                                    modify_2nd_order_terms,
                                    logs_per_epoch=logs_per_epoch,
                                    device=device)
        return train
    return training_fn


if __name__ == '__main__':
    max_blocks = [1, 2, 4, 16] #, 8, 32, 64, 128, 256, 512]
    seeds = range(10)

    # run experiments
    # ---------------
    for block in max_blocks:
        labels = [
                  'CG (GGN)',
                  'CG (PCH, abs)',
                  #'CG (PCH, clip)',
                 ]
        experiments = [
                       # 1) Generalized Gauss-Newton curve
                       cifar10_cgnewton_train_fn('zero', block),
                       # 2) BDA-PCH curve
                       cifar10_cgnewton_train_fn('abs', block),
                       # 3) alternative BDA-PCH curve
                       # cifar10_cgnewton_train_fn('clip', block),
                      ]


        metric_to_files = None
        for train_fn in experiments:
            runner = TrainingRunner(train_fn)
            runner.run(seeds)
