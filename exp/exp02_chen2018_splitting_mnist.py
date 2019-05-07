"""
Experiments performed in Chen et al.: BDA-PCH, figure 1,
with different parameter splitting.

Link to the reference:
    https://arxiv.org/pdf/1802.06502v2.pdf
"""

import torch
from torch.nn import CrossEntropyLoss
from os import path
from collections import OrderedDict
from exp.models.chen2018 import original_mnist_model
from exp.loading.load_mnist import MNISTLoader
from exp.training.second_order import SecondOrderTraining
from exp.training.runner import TrainingRunner
from exp.utils import (directory_in_data, dirname_from_params)
from bpexts.optim.cg_newton import CGNewton
from bpexts.hbp.parallel.sequential import HBPParallelSequential

# global hyperparameters
batch = 500
epochs = 20
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    run_name = dirname_from_params(
        opt='cgn',
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
    max_blocks = [1, 2, 4, 16]  # , 8, 32, 64, 128, 256, 512]
    seeds = range(10)

    labels = [
        # 'GGN',
        'PCH, abs',
        # 'PCH, clip',
    ]
    modify_2nd_order_terms = [
        # 1) GGN, different splittings
        # 'hzero',
        # 2) PCH, different splittings
        'abs',
        # 3) PCH alternative, different splittings
        # 'clip'
    ]

    # expand the labels for the experiments
    labels_expanded = []
    mod2nd_expanded = []
    blocks_expanded = []
    for label, mod2nd in zip(labels, modify_2nd_order_terms):
        labels_expanded += len(max_blocks) * [label]
        mod2nd_expanded += len(max_blocks) * [mod2nd]
        blocks_expanded += max_blocks

    experiments = [
        mnist_cgnewton_train_fn(mod2nd, blocks)
        for blocks, mod2nd in zip(blocks_expanded, mod2nd_expanded)
    ]

    def run():
        """Run the experiments."""
        for train_fn in experiments:
            runner = TrainingRunner(train_fn)
            runner.run(seeds)

    def result_files():
        """Merge runs and return files of the merged data."""
        filenames = OrderedDict()
        for label in labels:
            filenames[label] = OrderedDict()
        for label, block, train_fn in zip(labels_expanded, blocks_expanded,
                                          experiments):
            runner = TrainingRunner(train_fn)
            m_to_f = runner.merge_runs(seeds)
            filenames[label][block] = m_to_f
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
        corresponding to the block splitting which itself contains a dictionary
        with metric and filename of the metric data
        respectively.

        file_list = files[label][max_blocks][metric]
    """
    try:
        return main(run_experiments=False)
    except Exception as e:
        print("An error occured. Maybe try to re-run the experiments.")
        raise e


if __name__ == '__main__':
    main()
