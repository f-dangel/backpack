"""Run experiments with different seeds."""

import torch
import glob
import pandas
from bpexts.utils import set_seeds
from os import path, listdir
from warnings import warn


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
        """Run training procedure with different seeds, extract to CSV.

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
            sub = self._seed_directory_name(seed)
            if path.isdir(training.logger_subdir_path(sub)):
                warn('Logging directory {} already exists. Skipping'
                     ' run'.format(training.logger_subdir_path(sub)))
            else:
                training.point_logger_to_subdir(sub)
                training.run()
                training.logger.extract_logdir_to_csv()

    def merge_and_average_runs(self, seeds):
        """Convert runs to pandas and average over runs."""
        training = self.training_fn()
        seed_dirs = [training.logger_subdir_path(
            self._seed_directory_name(seed)) for seed in seeds] 


        # find all .csv files in the seed directories
        metrics = [path.basename(file)
                   for seed_dir in seed_dirs
                   for file in glob.glob(path.join(seed_dir, '*.csv'))]
        metrics = set(metrics)
        print('Found metrics {}'.format(metrics))

        for metric in metrics:
            files = [path.join(seed_dir, metric) for seed_dir in seed_dirs]
            pd = []
            # raises FileNotFoundError if one .csv file does not exist
            # no wall time
            for file, seed in zip(files, seeds):
                rename_dict = {
                               'step': 'step', 
                               'value' : 'seed{}'.format(seed)
                               }
                df = pandas.read_csv(file)[['step', 'value']]
                df = df.rename(index=str, columns=rename_dict)
                pd.append(df)

            joined = None
            while pd:
                pd_item = pd.pop()
                joined = pd_item if joined is None\
                         else joined.merge(pd_item, on=['step'])
            print(joined)

            # check for NaNs
            num_nans = joined.isnull().sum().sum()
            if num_nans != 0:
                raise self.InconsistentCSVError('Found {} NaN values'
                                                .format(num_nans))


            # computing mean and stddev
            data_idx = ['seed{}'.format(seed) for seed in seeds]
            joined['mean'] = joined[data_idx].mean(axis=1)
            joined['std'] = joined[data_idx].std(axis=1)

            print(joined)



            save_to = path.join(self._average_directory(), metric)
            print('Saving to {}'.format(save_to))
            joined.to_csv(save_to, index=False)

    class InconsistentCSVError(Exception):
        """Exception raised if CSV data is inconsistent."""
        pass

    def _average_directory(self):
        """Return the directory where the averages will be stored."""
        return self.training_fn().logdir

    @staticmethod
    def _seed_directory_name(seed):
        """Return name of subdirectory for a certain seed.

        Parameters:
        -----------
        seed : (int)
            Random seed of the run
        """
        return 'seed{}'.format(seed)

