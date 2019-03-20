Backpropagation extensions (bpexts)
###################################

PyTorch extension to compute additional quantities such as

-   Hessian blocks
 
-   batch gradients

by backpropagation for employment in 

-   2nd-order and
 
-   variance-adapted

optimization methods.


Installation
############

- Clone the repository

  .. code:: console

    $ git clone https://github.com/f-dangel/bpexts.git

- Change into the directory

  .. code:: console

    $ cd bpexts/

- Install dependencies and ``bpexts``

  .. code:: console

    $ pip3 install requirements.txt

    $ pip3 install .

  You should now be able to ``import bpexts`` in a ``pyhon3`` session.


Related papers (reproducing experiments)
########################################

- Dangel, F. and Hennig, P.: `A Modular Approach to Block-diagonal Hessian Approximations for Second-order Optimization <https://arxiv.org/abs/1902.01813>`_ (2019)

  - The work presents an extended backpropagation procedure, referred to as *Hessian backpropagation (HBP)*,
    for computing curvature approximations of feedforward neural networks.

  - To **reproduce the experiment** (Figure 5) in the paper, we recommend using the script in ``examples/2019_02_dangel_hbp/``.
    A step-by-step instruction is given in the README file in ``examples/2019_02_dangel_hbp/``.




Developer notes
###############

- **(Optional)** Run tests before committing: Copy the ``pre-commit`` file to your ``.git/hooks/`` directory.

- Virtual environment (assuming you are in the top directory of the repository)

  - Set up a virtual environment with

    .. code:: console

      $ virtualenv --python=/usr/bin/python3 .venv

  - Activate it

    .. code:: console

      $ source .venv/bin/activate

  - Install dependencies (also these for development/experiments)

    .. code:: console

      $ pip3 install -r ./requirements.txt

      $ # optional

      $ pip3 install -r ./requirements_exp.txt

  - Install the library (in editable mode)

    .. code:: console

      pip3 install --editable .

  - Deactivate the virtual environment by typing

    .. code:: console

      deactivate

- Run tests manually

  .. code:: console

    $ chmod u+x ./pre-commit

    $ ./pre-commit

    $ # alternative

    $ pytest -v bpexts

    $ pytest -v exp
