"""Experiments performed in Chen et al.: BDA-PCH, figure 1.

Link to the reference:
    https://arxiv.org/pdf/1802.06502v2.pdf
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from os import path
from models_chen2018 import original_mnist_model, separated_mnist_model
from load_mnist import MNISTLoader
from training import (FirstOrderTraining,
                      SecondOrderTraining)
from utils import (directory_in_data,
                   dirname_from_params,
                   tensorboard_instruction,
                   run_directory_exists)
import enable_import_bpexts
from bpexts.optim.cg_newton import CGNewton


if __name__ == '__main__':

    # ---------
    # MNIST SGD
    # ---------
    device = torch.device('cuda:0' if torch.cuda.is_available()
                          else 'cpu')

    # hyper parameters
    # ----------------
    batch = 500
    epochs = 30
    lr = 0.1
    momentum = 0.9

    # logging directory
    # -----------------
    directory_name = 'exp01_reproduce_chen_figures/mnist'
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
        # set up training and run
        model = original_mnist_model()
        # NOTE: Important line, deactivate extension hooks/buffers!
        model.disable_exts()
        loss_function = CrossEntropyLoss()
        data_loader = MNISTLoader(train_batch_size=batch)
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

    # --------------
    # MNIST CGNewton
    # --------------
    device = torch.device('cuda:0' if torch.cuda.is_available()
                          else 'cpu')

    # hyper parameters
    # ----------------
    batch = 500
    epochs = 30
    lr = 0.1
    alpha = 0.02
    modify_2nd_order_terms = 'abs'
    cg_maxiter = 50
    cg_tol = 0.1
    cg_atol = 0

    # logging directory
    # -----------------
    directory_name = 'exp01_reproduce_chen_figures/mnist'
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
        model = original_mnist_model()
        # works, scaling problem in Jacobian above
        # model = separated_mnist_model()
        loss_function = CrossEntropyLoss()
        data_loader = MNISTLoader(train_batch_size=batch)
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
