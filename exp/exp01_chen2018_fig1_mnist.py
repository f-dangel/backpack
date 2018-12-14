"""Experiments performed in Chen et al.: BDA-PCH, figure 1.

Link to the reference:
    https://arxiv.org/pdf/1802.06502v2.pdf
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from os import path
from models_chen2018 import original_mnist_model
from load_mnist import MNISTLoader
from training import Training
from utils import (directory_in_data,
                   dirname_from_params,
                   tensorboard_instruction,
                   run_directory_exists)


if __name__ == '__main__':

    # ---------
    # MNIST SGD
    # ---------

    # hyper parameters
    # ----------------
    batch = 500
    epochs = 30
    lr = 0.1
    momentum = 0.9
    device = torch.device('cuda:0' if torch.cuda.is_available()
                          else 'cpu')

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
        # setting up training and run
        model = original_mnist_model()
        model.disable_exts()
        loss_function = CrossEntropyLoss()
        data_loader = MNISTLoader(train_batch_size=batch)
        optimizer = SGD(model.parameters(),
                        lr=lr,
                        momentum=momentum)
        # initialize training
        train = Training(model,
                         loss_function,
                         optimizer,
                         data_loader,
                         logdir)
        train.run(num_epochs=epochs,
                  device=device)
