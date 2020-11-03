from os import path

from setuptools import find_packages, setup

# META
##############################################################################
AUTHORS = "F. Dangel, F. Kunstner"
NAME = "backpack-for-pytorch"
PACKAGES = find_packages()

DESCRIPTION = "BackPACK: Packing more into backprop"
LONG_DESCR = """
    BackPACK is built on top of PyTorch.
    It efficiently computes quantities other than the gradient.

    Website: https://backpack.pt
    Code: https://github.com/f-dangel/backpack
    Documentation: https://readthedocs.org/projects/backpack/
    Bug reports & feature requests: https://github.com/f-dangel/backpack/issues
    """

VERSION = "1.2.0"
URL = "https://github.com/f-dangel/backpack"
LICENSE = "MIT"

# DEPENDENCIES
##############################################################################
REQUIREMENTS_FILE = "requirements.txt"
REQUIREMENTS_PATH = path.join(path.abspath(__file__), REQUIREMENTS_FILE)

with open(REQUIREMENTS_FILE) as f:
    requirements = f.read().splitlines()

setup(
    author=AUTHORS,
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCR,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    url=URL,
    license=LICENSE,
    packages=PACKAGES,
    zip_safe=False,
    python_requires=">=3.6",
)
