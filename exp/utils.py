"""Utility functions for running experiments."""

from os import path
from warnings import warn


def directory_in_data(dir_name):
    """Return path in data folder with given name."""
    # directory to log quantities : ../dat
    parent_dir = path.dirname(path.dirname(path.realpath(__file__)))
    return path.join(parent_dir, 'dat', dir_name)


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
