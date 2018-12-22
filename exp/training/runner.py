"""Run experiments with different seeds."""

import torch
from bpexts.utils import set_seeds

class TrainingRunner():
    """Run experiments with different seeds."""
    def __init__(self, training_fn):
        """
        Parameters:
        -----------
        training_fn : (function)
            Function that creates an instance of `Training` (see
            `training.py`
        """
        self.training_fn = training_fn


    def run(self, seeds):
        """Run training procedure with different seeds.

        Parameters:
        -----------
        seeds : (list(int))
            List with init values for random seed
        num_epochs : (int)
            Number of epochs to train on
        device : (torch.Device)
           Device to run the training on
        logs_per_epoch : (int)
            Number of datapoints written into logdir per epoch
        """
        for seed in seeds:
            set_seeds(seed)
            training = self.training_fn()
            training.point_logger_to('seed{}'.format(seed))
            training.run()
