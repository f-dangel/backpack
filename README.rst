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

TODO


Related papers (reproducing experiments)
########################################

- Dangel, F. and Hennig, P.: `A Modular Approach to Block-diagonal Hessian Approximations for Second-order Optimization <https://arxiv.org/abs/1902.01813>`_ (2019)

  - The work presents an extended backpropagation procedure, referred to as *Hessian backpropagation (HBP)*,
    for computing curvature approximations of feedforward neural networks.

  - To **reproduce the experiment** (Figure 5) in the paper, we recommend using the script in ``examples/2019_02_dangel_hbp/``.
    A step-by-step instruction is given in the README file in ``examples/2019_02_dangel_hbp/``.


TODO: Clean up below

developer notes
===============

-  Please copy the ``pre-commit`` file to your ``.git/hooks/``
   directory. It will run tests before accepting commits.

virtualenv notes
================

-  Change into the repository directory and set up a virtualenv with
   *Python 3*:

   .. code:: console

       virtualenv --python=/usr/bin/python3 .venv

-  Once you have set up the virtual environment, it can be activated by

.. code:: console

    source .venv/bin/activate

-  After activating the environment, install dependencies

.. code:: console

    pip3 install -r ./requirements.txt

-  Additional dependencies for benchmarks, tests and experiments

.. code:: console

    pip3 install -r ./requirements_dev.txt

-  Install the library (in editable mode)

.. code:: console

    pip3 install --editable .

-  (Optional) run tests using the ``pre-commit`` script

.. code:: console

    chmod u+x ./pre-commit
    ./pre-commit

-  Deactivate the virtual environment by typing

.. code:: console

    deactivate
