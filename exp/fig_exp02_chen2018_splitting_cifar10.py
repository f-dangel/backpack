"""Plot experiment figure of ICML 2019 paper."""


from os import path, makedirs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from palettable.colorbrewer.sequential import YlGnBu_5, YlOrRd_5
from .plotting.plotting import OptimizationPlot
from .exp01_chen2018_fig2_cifar10 import filenames as exp01_files
from .exp02_chen2018_splitting_cifar10 import filenames as exp02_files
from .exp02_chen2018_splitting_cifar10 import dirname as exp02_dirname
from .utils import directory_in_fig


# define colors
sgd_color = YlOrRd_5.mpl_colors[2]
block_colors = YlGnBu_5.mpl_colors[1:][::-1]
colors = [sgd_color] + block_colors
# set color cycle
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=colors)


def plot():
    """Produce figure of block splitting on CIFAR-10."""
    # load data from exp01
    files01 = exp01_files()
    # load data filenames and set output directory
    files02 = exp02_files()
    fig_dir = directory_in_fig(exp02_dirname)
    # find metrics
    metrics = set(m for (_, exp) in files01.items() for m in exp.keys())

    fig_dir = directory_in_fig(exp02_dirname)

    for metric in metrics:
        for fig_sub, block_dict in files02.items():
            # output file
            this_fig_dir = path.join(fig_dir, fig_sub)
            out_file = path.join(this_fig_dir, metric)
            makedirs(this_fig_dir, exist_ok=True)

            # create figure
            plt.figure()
            # plot SGD
            plot_labels = ['SGD']
            plot_files = [files01['SGD'][metric]]
            OptimizationPlot.create_standard_plot('epoch',
                                                  metric.replace('_', ' '),
                                                  plot_files,
                                                  plot_labels,
                                                  # scale by training set
                                                  scale_steps=50000)

            # collect plots for block splitting
            plot_labels = []
            plot_files = []
            for b in sorted(block_dict.keys()):
                plot_files.append(block_dict[b][metric])
                plot_labels.append('CG, {} block{}'
                                   .format(b, 's' * (b != 1)))
            # plot block splitting
            OptimizationPlot.create_standard_plot('epoch',
                                                  metric.replace('_', ' '),
                                                  plot_files,
                                                  plot_labels,
                                                  # scale by training set
                                                  scale_steps=50000)
            plt.legend()
            OptimizationPlot.save_as_tikz(out_file)
            OptimizationPlot.post_process(out_file)


if __name__ == '__main__':
    plot()
