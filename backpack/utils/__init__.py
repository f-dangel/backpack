"""Contains utility functions."""

from pkg_resources import get_distribution, packaging

TORCH_VERSION = packaging.version.parse(get_distribution("torch").version)
VERSION_1_8_0 = packaging.version.parse("1.8.0")
TORCH_VERSION_HIGHER_THAN_1_8_0 = VERSION_1_8_0 <= TORCH_VERSION
