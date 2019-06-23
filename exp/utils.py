"""Utility functions for running experiments.

* directory to log quantities : ../dat
* directory to store figures : ../fig
"""

from os import path
from warnings import warn

parent_dir = path.dirname(path.dirname(path.realpath(__file__)))


def directory_in_data(dir_name):
    """Return path in data folder with given name."""
    return path.join(parent_dir, 'dat', dir_name)


def directory_in_fig(dir_name):
    """Return path in fig folder with given name."""
    return path.join(parent_dir, 'fig', dir_name)


def dirname_from_params(**kwargs):
    """Concatenate key, value pairs alphabetically, split by underscore."""
    ordered = sorted(kwargs.items())
    words = ['_'.join([key, str(value)]) for key, value in ordered]
    return '_'.join(words)


def run_directory_exists(logdir):
    """Return warning: Run directory exists, will be skipped."""
    if path.isdir(logdir):
        warn('\nLogging directory already exists:\n{}\n'
             'It is likely that this run will be skipped.\n'.format(logdir))
        return True
    return False


def centered_list(sequence):
    """Return a list iterating from the center to the outside.

    Parameters:
    -----------
    sequence : list, tuple
        Object supporting indexing

    Examples:
    ---------
    >>> centered_list([1, 2, 3, 4])
    [3, 2, 4, 1]
    >>> centered_list([1, 2, 3, 4, 5])
    [3, 2, 4, 1, 5]
    """
    center_idx = max(len(sequence) // 2, 0)

    def is_valid(idx):
        return 0 <= idx <= len(sequence) - 1

    # fill the list
    centered_sequence = [sequence[center_idx]]
    for shift in range(1, len(sequence) // 2 + 1):
        if is_valid(center_idx - shift):
            centered_sequence.append(sequence[center_idx - shift])
        if is_valid(center_idx + shift):
            centered_sequence.append(sequence[center_idx + shift])
    return centered_sequence
