"""Plot experiment of training c4d3 with sigmoid activations."""

from os import path, makedirs
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from palettable.colorbrewer.sequential import YlGnBu_5, YlOrRd_5
from exp.plotting.plotting import OptimizationPlot
from exp.exp08_c4d3_optimization_cvp import filenames as exp08_cvp_files
from exp.exp08_c4d3_optimization_sgd import filenames as exp08_sgd_files
from exp.exp08_c4d3_optimization_adam import filenames as exp08_adam_files
from exp.exp08_c4d3_optimization_sgd import parent_dir as exp08_dirname
from exp.utils import directory_in_fig

# define colors
sgd_color = YlOrRd_5.mpl_colors[2]
adam_color = YlOrRd_5.mpl_colors[4]
cvp_colors = YlGnBu_5.mpl_colors[1:][::-1]
colors = [sgd_color] + [adam_color] + cvp_colors
# set color cycle
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=colors)


def plot():
    # load data from exp08
    files_sgd = exp08_sgd_files()
    files_cvp = exp08_cvp_files()
    files_adam = exp08_adam_files()
    # set output directory
    fig_dir = directory_in_fig(exp08_dirname)
    # find metrics
    metrics = set(m for (_, exp) in files_sgd.items() for m in exp.keys())

    for metric in metrics:
        for fig_sub, block_dict in files_sgd.items():
            # output file
            this_fig_dir = fig_dir  # path.join(fig_dir, fig_sub)
            out_file = path.join(this_fig_dir, metric)
            makedirs(this_fig_dir, exist_ok=True)

            # create figure
            plt.figure()
            # plot sgd
            plot_labels = ['SGD']
            plot_files = [files_sgd['SGD'][metric]]
            OptimizationPlot.create_standard_plot(
                'epoch',
                metric.replace('_', ' '),
                plot_files,
                plot_labels,
                # scale by training set
                scale_steps=50000)

            # plot Adam
            plot_labels = ['Adam']
            plot_files = [files_adam['Adam'][metric]]
            OptimizationPlot.create_standard_plot(
                'epoch',
                metric.replace('_', ' '),
                plot_files,
                plot_labels,
                # scale by training set
                scale_steps=50000)

            # collect plots for different curvature matrices
            plot_labels = [
                r'GGN, $\alpha_1$',
                r'GGN, $\alpha_2$',
                # 'PCH-abs1',
                'PCH-abs',
                'PCH-clip',
            ]
            # plot_labels = ['CG, GGN', 'CG, PCH (abs)', 'CG, PCH (clip)']
            plot_files = [files_cvp[l][metric] for l in plot_labels]
            idx = [2, 3, 0, 1]
            plot_files = [plot_files[i] for i in idx]
            plot_labels = [plot_labels[i] for i in idx]
            # plot CGN curvature matrices
            OptimizationPlot.create_standard_plot(
                'epoch',
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
