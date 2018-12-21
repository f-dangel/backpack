"""Experiments performed in Chen et al.: BDA-PCH, figure 2.

Link to the reference:
    https://arxiv.org/pdf/1802.06502v2.pdf
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from os import path
from models_chen2018 import original_cifar10_model
from load_cifar10 import CIFAR10Loader
from training import (FirstOrderTraining,
                      SecondOrderTraining)
from utils import (directory_in_data,
                   dirname_from_params,
                   tensorboard_instruction,
                   run_directory_exists)
import enable_import_bpexts
from bpexts.optim.cg_newton import CGNewton


def cifar10_sgd():
    """
    # -----------
    # CIFAR10 SGD
    # -----------

    Run will be skipped if logging directory already exists.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available()
                          else 'cpu')

    # hyper parameters
    # ----------------
    batch = 500
    epochs = 100
    lr = 0.1
    momentum = 0.9

    # logging directory
    # -----------------
    directory_name = 'exp01_reproduce_chen_figures/cifar10'
    data_dir = directory_in_data(directory_name)
    print(tensorboard_instruction(data_dir))
    # directory of run
    run_name = dirname_from_params(opt='sgd',
                                   batch=batch,
                                   lr=lr,
                                   mom=momentum)
    logdir = path.join(data_dir, run_name)

    # training procedure
    # ------------------
    if not run_directory_exists(logdir):
        # setting up training and run
        model = original_cifar10_model()
        model.disable_exts()
        loss_function = CrossEntropyLoss()
        data_loader = CIFAR10Loader(train_batch_size=batch)
        optimizer = SGD(model.parameters(),
                        lr=lr,
                        momentum=momentum)
        # initialize training
        train = FirstOrderTraining(model,
                                   loss_function,
                                   optimizer,
                                   data_loader,
                                   logdir)
        train.run(num_epochs=epochs,
                  device=device)


def cifar10_cgnewton(modify_2nd_order_terms):
    """
    # ----------------
    # CIFAR-10CGNewton
    # ----------------

    Run will be skipped if logging directory already exists.

    Parameters:
    -----------
    modify_2nd_order_terms : (str)
        Strategy for treating 2nd-order effects of module functions:
        * `'zero'`: Yields the Generalizes Gauss Newton matrix
        * `'abs'`: BDA-PCH approximation
        * `'clip'`: Different BDA-PCH approximation
    """
    # Requires ~3GB of RAM
    # device = torch.device('cuda:0' if torch.cuda.is_available()
    #                       else 'cpu')
    device = torch.device('cpu')

    # hyper parameters
    # ----------------
    batch = 500
    epochs = 100
    lr = 0.1
    alpha = 0.02
    cg_maxiter = 50
    cg_tol = 0.1
    cg_atol = 0

    # logging directory
    # -----------------
    directory_name = 'exp01_reproduce_chen_figures/cifar10'
    data_dir = directory_in_data(directory_name)
    print(tensorboard_instruction(data_dir))
    # directory of run
    run_name = dirname_from_params(opt='cgn',
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
    if not run_directory_exists(logdir):
        # set up training and run
        model = original_cifar10_model()
        loss_function = CrossEntropyLoss()
        data_loader = CIFAR10Loader(train_batch_size=batch)
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
                                    logdir)
        train.run(num_epochs=epochs,
                  modify_2nd_order_terms=modify_2nd_order_terms,
                  device=device)


if __name__ == '__main__':
    # 1) SGD curve
    cifar10_sgd()

    # 2) Jacobian curve
    cifar10_cgnewton('zero')

    # 3) BDA-PCH curve
    cifar10_cgnewton('abs')

    # 4) alternative BDA-PCH curve
    cifar10_cgnewton('clip')
