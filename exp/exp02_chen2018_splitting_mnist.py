"""
Experiments performed in Chen et al.: BDA-PCH, figure 1,
with different parameter splitting.

Link to the reference:
    https://arxiv.org/pdf/1802.06502v2.pdf
"""

import torch
from torch.nn import CrossEntropyLoss
from os import path, makedirs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .models.chen2018 import original_mnist_model
from .loading.load_mnist import MNISTLoader
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
epochs = 20
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
dirname = 'exp02_chen_splitting/mnist'
data_dir = directory_in_data(dirname)
fig_dir = directory_in_fig(dirname)
logs_per_epoch = 5


def mnist_cgnewton_train_fn(modify_2nd_order_terms, max_blocks):
    """Create training instance for MNIST CG experiment.

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
    def train_fn():
        """Training function setting up the train instance."""
        # set up training and run
        model = original_mnist_model()
        # split into parallel modules
        model = HBPParallelSequential(max_blocks, *list(model.children()))
        loss_function = CrossEntropyLoss()
        data_loader = MNISTLoader(train_batch_size=batch,
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
    return train_fn


if __name__ == '__main__':
    max_blocks = [1, 2, 4, 16] #, 8, 32, 64, 128, 256, 512]
    seeds = range(10)

    titles = [
              'GGN',
              'PCH, abs',
              'PCH, clip',
             ]
    fig_subdirs = [
                   'GGN',
                   'PCH-abs',
                   'PCH-clip'
                  ]
    modify_2nd_order_terms = [
                              # 1) GGN, different splittings
                              'zero',
                              # 2) PCH, different splittings
                              'abs',
                              # 3) PCH alternative, different splittings
                              'clip'
                              ]

    for title, mod2nd, fig_sub in zip(titles,
                                      modify_2nd_order_terms,
                                      fig_subdirs):
        # dict of dicts stores same metrics for different blocks
        metric_to_file_for_blocks = {}

        for blocks in max_blocks:
            train_fn = mnist_cgnewton_train_fn(mod2nd, blocks)

            # run experiments
            # ---------------
            runner = TrainingRunner(train_fn)
            runner.run(seeds)
            metric_to_files = runner.merge_runs(seeds)

            # initialize with empty dict for each metric if empty
            if metric_to_file_for_blocks == {}:
                for metric in metric_to_files.keys():
                    metric_to_file_for_blocks[metric] = {}
            # sort by blocks
            for metric, files in metric_to_files.items():
                metric_to_file_for_blocks[metric][blocks] = files

        # plotting
        # --------
        for metric, block_dict in metric_to_file_for_blocks.items():
            # output file
            this_fig_dir = path.join(fig_dir, fig_sub)
            out_file = path.join(this_fig_dir, metric)
            makedirs(this_fig_dir, exist_ok=True)
            # files for each metric with labels for blocks
            files, labels = [], []
            for b in sorted(block_dict.keys()):
                files.append(block_dict[b])
                labels.append('CG, {} block{}'.format(b, 's' if b != 1 else ''))
            # figure
            plt.figure()
            plt.title(title)
            OptimizationPlot.create_standard_plot('epoch',
                                                  metric.replace('_', ' '),
                                                  files,
                                                  labels,
                                                  plot_std=False,
                                                  # scale by training set
                                                  scale_steps=60000)
            plt.legend()
            # fine tuning
            if '_loss' in metric:
                plt.ylim(bottom=-0.05, top=1)
            OptimizationPlot.save_as_tikz(out_file)
