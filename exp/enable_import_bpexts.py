"""Allow package bpexts located in parent directory to be imported."""

from os import path
from imp import load_source

__package_name = 'bpexts'
__package_dir = path.dirname(path.dirname(path.realpath(__file__)))
__package_path = path.join(__package_dir, __package_name, '__init__.py')
load_source(__package_name, __package_path)
