"""Plot the data computed in `exp01_chen2018_fig2_cifar10.py`."""


from os import path, makedirs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .plotting.plotting import OptimizationPlot
from .exp01_chen2018_fig2_cifar10 import filenames as exp01_files
from .exp01_chen2018_fig2_cifar10 import dirname as exp01_dirname
from .utils import directory_in_fig


def plot():
    """Reproduce figure 1 of Chen et al. (2018), MNIST."""
    # load data filenames and set output directory
    filenames = exp01_files()
    fig_dir = directory_in_fig(exp01_dirname)
    # find metrics and legend entries to plot
    plot_labels = filenames.keys()
    metrics = set(m for (_, exp) in filenames.items() for m in exp.keys())

    for metric in metrics:
        out_file = path.join(fig_dir, metric)
        makedirs(fig_dir, exist_ok=True)
        plot_files = [filenames[label][metric] for label in plot_labels]
        # figure
        plt.figure()
        OptimizationPlot.create_standard_plot('epoch',
                                              metric.replace('_', ' '),
                                              plot_files,
                                              plot_labels,
                                              # scale by training set
                                              scale_steps=50000)
        plt.legend()
        OptimizationPlot.save_as_tikz(out_file)


if __name__ == '__main__':
    plot()
