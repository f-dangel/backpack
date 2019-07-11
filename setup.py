from os import path
from setuptools import setup

# META
##############################################################################
AUTHORS = "F. Dangel, F. Künstner"
NAME = "BackPACK"
# TODO: Change into backpack
PACKAGES = ["bpexts"]

DESCRIPTION = r"""BACKpropagation PACKage - A backpack for PyTorch that
extends the backward pass of feedforward networks to compute quantities
beyond the gradient.
""".replace("\n", " ")

VERSION = "0.1"
# TODO: Change into backpack
URL = "https://github.com/f-dangel/bpexts"
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
    install_requires=requirements,
    url=URL,
    license=LICENSE,
    packages=PACKAGES,
    zip_safe=False,
)
