Reproduce experiments
#####################

This file provides a straightforward explanation on how to reproduce the experiment from

- Dangel, F. and Hennig, P.: `A Modular Approach to Block-diagonal Hessian Approximations for Second-order Optimization <https://arxiv.org/abs/1902.01813>`_ (2019)

We provide an executable ``run_experiments.sh`` script. It runs the experiments by setting up a virtual environment, that is removed again after execution.

Step 1/3: Preliminaries
***********************

- Make sure you have `python3` installed (we tested on ``Python 3.5.2`` and ``Python 3.6.7``).

- Make sure you have an installation of ``virtualenv``

  .. code:: console
              
     $ pip3 install virtualenv

Step 2/3: Preparing the files
*****************************

- Open a terminal 

- Clone the repository

  .. code:: console
              
     $ git clone https://github.com/f-dangel/bpexts.git

- Change into the repository directory

  .. code:: console

     $ cd bpexts/

- Check out the version of the code that was used to produce the original experiment

  .. code:: console

     $ git checkout 2019_03_HBP_reproduce

- Copy the run script to the repository level

  .. code:: console

     $ cp examples/2019_02_dangel_hbp/run_experiments.sh .

Step 3/3: Run experiment
************************

**Note:** If you do not want to execute the script on your machine (method 1), you can also run it inside of a `docker` container (method 2). However, this solution runs experiments only on the CPU, even if a GPU is available.


Method 1: Run script
====================

- Make the script executable:

  .. code:: console
    
      $ chmod u+x run_experiments.sh

- **(Optional)** Read the explanations in ``run_experiments.sh``. If you want to change the number of jobs that are being processed in parallel, adapt the line

  .. code:: console

      NUMJOBS=1

- Run the script:

  .. code:: console
    
     $ ./run_experiments.sh

**Congratulations!** The script will give you a rough estimate of the remaining compute time and prompt you to the directory of the figures.


Method 2: Run inside a ``docker`` container
===========================================

- Make sure your have ``docker`` installed

- Copy the ``Dockerfile``

  .. code:: console
        
     $ cp examples/2019_02_dangel_hbp/Dockerfile .

- Build the container named ``2019_02_dangel_hbp`` using the provided ``Dockerfile``:

  .. code:: console
        
     $ docker build -t 2019_02_dangel_hbp .

- Launch the container and start a ``bash`` session within:

  .. code:: console

     $ docker run -it 2019_02_dangel_hbp bash

  You will now see a command prompt similar to the one below:
    
  .. code:: console
    
     root@2a756b23e4:/#

- Change into the ``home/`` directory that holds copies of the repository

  .. code:: console
    
     $ cd home/

- Perform the steps described in **Method 1**, but in the ``docker`` container

- If you want to copy the results to your machine, ``docker`` provides a way for doing so. Check out
    
  - https://stackoverflow.com/questions/22049212/copying-files-from-docker-container-to-host
        
  for more details.

