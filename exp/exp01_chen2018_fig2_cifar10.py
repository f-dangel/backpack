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

if __name__ == '__main__':

    # -----------
    # CIFAR10 SGD
    # -----------
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
