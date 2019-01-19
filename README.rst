backprop-extended
=================

PyTorch extension to compute additional quantities such as

-   Hessian blocks
 
-   batch gradients

by backpropagation for employment in 

-   2nd-order and
 
-   variance-adapted

optimization methods..


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
