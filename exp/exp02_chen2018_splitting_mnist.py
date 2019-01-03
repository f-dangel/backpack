"""
Experiments performed in Chen et al.: BDA-PCH, figure 1,
with different parameter splitting.

Link to the reference:
    https://arxiv.org/pdf/1802.06502v2.pdf
"""

import torch
from torch.nn import CrossEntropyLoss
from os import path, makedirs
from .models.chen2018 import original_mnist_model
from .loading.load_mnist import MNISTLoader
from .training.second_order import SecondOrderTraining
from .training.runner import TrainingRunner
#from .plotting.plotting import OptimizationPlot
from .utils import (directory_in_data,
                    directory_in_fig,
                    dirname_from_params)
from bpexts.optim.cg_newton import CGNewton
from bpexts.hbp.parallel.parallel_sequential\
        import HBPParallelSequential


# global hyperparameters
batch = 500
epochs = 20
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dirname = 'exp02_chen_splitting/mnist'
data_dir = directory_in_data(dirname)
fig_dir = directory_in_fig(dirname)
logs_per_epoch = 5


def mnist_cgnewton_train_fn(modify_2nd_order_terms, num_blocks,
                            # input_hessian_mode
                            ):
    """Create training instance for MNIST CG experiment.

    Trainable parameters (weights and bias) will be split into
    subplots during optimization.

    Parameters:
    -----------
    modify_2nd_order_terms : (str)
        Strategy for treating 2nd-order effects of module functions:
        * `'zero'`: Yields the Generalizes Gauss Newton matrix
        * `'abs'`: BDA-PCH approximation
        * `'clip'`: Different BDA-PCH approximation
    num_blocks : (int)
        * Split parameters per layer into subblocks during
          optimization
    input_hessian_mode : (str)
        `"exact"` or `"blockwise"`. Strategy for computing the
        Hessian with respect to a parallel layer`s input
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
                                   blocks=num_blocks,
                                   # modhin=input_hessian_mode
                                   )
    logdir = path.join(data_dir, run_name)

    # training procedure
    # ------------------
    def train_fn():
        """Training function setting up the train instance."""
        # set up training and run
        model = original_mnist_model()
        # split into parallel modules
        model = HBPParallelSequential.from_sequential(model)
        model = model.split_into_blocks(num_blocks)
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
    num_blocks = [1]  # , 2, 3, 4, 5, 256, 512]
    seeds = range(1)

    labels = [
              # 'CG (GGN)',
              'CG (PCH, abs)',
              # 'CG (PCH, clip)',
             ]
    experiments = [
                   # 1) Jacobian curve
                   # mnist_cgnewton_train_fn('zero'),
                   # 2) BDA-PCH curve
                   mnist_cgnewton_train_fn('abs', 10),
                   # 3) alternative BDA-PCH curve
                   # mnist_cgnewton_train_fn('clip'),
                  ]

    # run experiments
    # ---------------
    metric_to_files = None
    for train_fn in experiments:
        runner = TrainingRunner(train_fn)
        runner.run(seeds)
        """
        m_to_f = runner.merge_runs(seeds)
        if metric_to_files is None:
            metric_to_files = {k: [v] for k, v in m_to_f.items()}
        else:
            for key, value in m_to_f.items():
                metric_to_files[key].append(value)
        """

    """
    # plotting
    # --------
    for metric, files in metric_to_files.items():
        out_file = path.join(fig_dir, metric)
        makedirs(fig_dir, exist_ok=True)
        # figure
        plt.figure()
        OptimizationPlot.create_standard_plot('epoch',
                                              metric.replace('_', ' '),
                                              files,
                                              labels,
                                              # scale by training set
                                              scale_steps=60000)
        plt.legend()
        # fine tuning
        if '_loss' in metric:
            plt.ylim(bottom=-0.05, top=1)
        OptimizationPlot.save_as_tikz(out_file)
    """
