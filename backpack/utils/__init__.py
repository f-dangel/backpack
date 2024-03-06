"""Contains utility functions."""

from pkg_resources import get_distribution, packaging

TORCH_VERSION = packaging.version.parse(get_distribution("torch").version)
TORCH_VERSION_AT_LEAST_1_13 = TORCH_VERSION >= packaging.version.parse("1.13")
