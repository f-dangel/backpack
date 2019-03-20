#!/bin/bash

#########################################################
# Generate the results shown in the experiment section. #
#########################################################
# Total run time (GPU): ~24 hours
# Memory requirements (GPU): 700 MB per job

# Modify if you want to run multiple jobs in parallel
# (1 to 4 should be reasonable depending on your machine)
NUMJOBS=1

# The figure will be generated in the code/fig directory.

# Overall progress can be monitored by opening the tensor-
# board link that is printed right before the experiments
# will be started.

# Running this script should not mess up any files outside
# of the directory that contains the script.

# It performs the following steps:
#    1) Set up virtual environment
#    2) Install dependencies and implementation of HBP
#    3) Run the experiments
#    4) Plot the results
#    5) Clean up, remove virtual environment

# Each of the 5 lines requires 10 runs over different seeds.
# With each run taking roughly 20-30min on a GPU, the total
# computation should be roughly about 24 hours.


#####################
# ! DO NOT MODIFY ! #
#####################


###########################################################
# kill tensorboard if it is still running
killall tensorboard
# change into directory
cd code
# clean up virtual environment and other files if last
# launch of the script failed or was cancelled
rm -rf .venv
rm -rf dat
rm -rf fig
###########################################################


###########################################################
# 1) virtual environment
printf "\n\n########################################\n"
printf "# CREATING VIRTUAL ENVIRONMENT (2 MIN) #\n"
printf "########################################\n\n"
virtualenv --python=/usr/bin/python3 .venv
# activate it
source .venv/bin/activate
###########################################################


###########################################################
# 2) requirements
printf "\n\n####################################\n"
printf "# INSTALLING REQUIREMENTS (10 MIN) #\n"
printf "####################################\n\n"
pip3 install -r ./requirements.txt
pip3 install -r ./requirements_exp.txt
pip3 install .
###########################################################


###########################################################
# 3) Experiments: monitor progress
# launch tensorboard in background
printf "\n\n###########################\n"
printf "# TENSORBOARD INSTRUCTION #\n"
printf "###########################\n\n"
tensorboard --logdir dat &
sleep 10s
###########################################################


###########################################################
# 3a) Run SGD experiments
(
printf "\n\n#################################\n"
printf "# RUNNING SGD EXPERIMENTS (4 H) #\n"
printf "#################################\n\n"
for i in `seq 1 $NUMJOBS`
do
    printf "\nStarting parallel process $i / $NUMJOBS\n"
    # wait 10min for download of CIFAR-10 in first call
    if  [ "$i" -eq "1" ]
    then
        printf "Downloading CIFAR-10:\n"
        python3 -m exp.exp01_chen2018_fig2_cifar10 &
        sleep 10m
    else
        python3 -m exp.exp01_chen2018_fig2_cifar10 &
        sleep 10s
    fi
done
# wait for jobs to finish
wait
)
###########################################################


###########################################################
# 3b) Run block-splitting CG experiments
(
printf "\n\n###########################################\n"
printf "# RUNNING CG SPLITTING EXPERIMENTS (20 H) #\n"
printf "###########################################\n\n"
for i in `seq 1 $NUMJOBS`
do
    printf "\nStarting parallel process $i / $NUMJOBS\n"
    python3 -m exp.exp02_chen2018_splitting_cifar10 &
    sleep 10s
done
# wait for jobs to finish
wait
)
###########################################################


###########################################################
# 4) Create figure
printf "\n\n###########################\n"
printf "# CREATING FIGURE (1 MIN) #\n"
printf "###########################\n\n"
python3 -m exp.fig_exp02_chen2018_splitting_cifar10
###########################################################


###########################################################
# 5) Clean up
# kill tensorboard
killall tensorboard
# delete virtual environment
deactivate
rm -rf .venv
###########################################################


printf "\n\nFind the figure in code/fig/\n\n\n"
