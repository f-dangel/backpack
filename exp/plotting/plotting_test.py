"""Integration test for standard optimization plots.

Illustrates how to set color schemes with better visibility
using the schemes provided by `palettable`."""

from ..utils import (directory_in_data,
                     directory_in_fig)
from .plotting import OptimizationPlot
import numpy as np
import pandas
from os import path, makedirs
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from palettable.colorbrewer.sequential import Blues_4


# set up directories for dummy data and figures
dirname = 'test_plotting'
data_dir = directory_in_data(dirname)
makedirs(data_dir, exist_ok=True)
fig_dir = directory_in_fig(dirname)
makedirs(fig_dir, exist_ok=True)
# filename of plot
plot_file = path.join(fig_dir, 'test_plot')
# filenames of the dummy .CSV files
filenames = [path.join(data_dir, f) for f in ['dummy_A.csv',
                                              'dummy_B.csv',
                                              'dummy_C.csv',
                                              'dummy_D.csv']]
# labels in plot
labels = [r'$\sin$', r'$\cos$', r'$\tanh$', r'$\exp(-x^2)$']
# dummy functions
functions = [np.sin, np.cos, np.tanh, lambda x: np.exp(-x**2)]


def create_dummy_data():
    """Create four dummy datasets.

    The data is stored in the corresponding `data_dir`.
    """
    num_steps = 50
    x = np.linspace(0, 5, num_steps)
    for filename, f in zip(filenames, functions):
        df = pandas.DataFrame()
        df['step'] = x
        df['mean'] = f(x)
        df['std'] = 0.1 * np.random.randn(num_steps)
        df.to_csv(filename, index=False)


def test_plotting():
    """Test creation of plots."""
    create_dummy_data()
    # plot and save
    plt.figure()
    # set color cycle
    matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
            color=Blues_4.mpl_colors)
    xlabel = r'$x$'
    ylabel = r'$f(x)$'
    OptimizationPlot.create_standard_plot(xlabel,
                                          ylabel,
                                          filenames,
                                          labels)
    plt.legend()
    OptimizationPlot.save_as_tikz(plot_file)
