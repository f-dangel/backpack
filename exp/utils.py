"""Utility functions for running experiments."""

from os import path
from imp import load_source


def enable_import_bpexts_without_installation():
    """Allow package bpexts located in parent directory to be imported."""
    package_name = 'bpexts'
    package_dir = path.dirname(
            path.dirname(path.realpath(__file__)))
    package_path = path.join(package_dir,
                             package_name,
                             '__init__.py')
    load_source(package_name, package_path)
