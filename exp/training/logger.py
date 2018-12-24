"""Class for handling logging for tensorboard and conversion to CSV."""

from tensorboard_logger import Logger as TensorboardLogger
from tensorboard.backend.event_processing.event_accumulator\
        import EventAccumulator
from os import path
import pandas


class Logger(TensorboardLogger):
    """Handle logging during experiment and extraction to csv.

    Allow for pointing the logger to different subdirectories.
    
    Private attributes:
    -------------------
    __logdir : (str)
        Main directory of the logger
    __subdir : (str)
        Subdirectory in main directory of the logger
    """
    def __init__(self, logdir, subdir=''):
        self.__logdir = logdir
        self.__subdir = subdir
        super().__init__(self.subdir_path(self.__logdir, self.__subdir))

    def point_to_subdir(self, subdir):
        """Point logger to subdirectory.

        Parameters:
        -----------
        subdir : (str)
            name of the subdirectory logging directory

        Returns:
        --------
        (Logger)
            New instance of logger pointing to the subdirectory
        """
        self.__init__(self.__logdir, subdir)

    @staticmethod
    def subdir_path(dir_, subdir):
        """Return Path of subdirectory in directory."""
        return path.join(dir_, subdir)

    def log_scalar_values(self, summary, step):
        """Log all key-value pairs in `summary` at `step`.

        Parameters:
        -----------
        summary : (dict)
            Dictionary with scalar items step : (int)
            Step for logging (must be int)
        """
        for key, value in summary.items():
            super().log_value(key, value, step)

    def print_tensorboard_instruction(self):
        """Print message on how to display results using tensorboard."""
        print('\nLogging data into {}\n\nUse tensorboard to'
              ' investigate the output by typing\n\t'
              ' tensorboard --logdir {}\n'.format(self.__logdir,
                                                  self.__logdir))

    def extract_logdir_to_csv(self):
        """Extract scalar quantities in current logdir to .csv file."""
        for tag, data in self._extract_logdir_to_pandas().items():
            filename = path.join(self.logdir, tag + '.csv')
            print('Save scalar to', filename, '...', end='')
            data.to_csv(filename, index=False)
            print('Successfull')

    def _extract_logdir_to_pandas(self):
        """Return pandas dataframes for all metrics in logdir.
        
        Returns:
        --------
        (dict(pandas.DataFrame))
            Dictionary with (key, value) pairs corresponding to the name
            and the data of the metric
        """
        return self._scalars_to_pandas(self.logdir)

    @classmethod
    def _scalars_to_pandas(cls, event_dir):
        """Extract scalar quantities in event directory to pandas DataFrame.

        Parameters:
        -----------
        event_dir : (str)
            Directory containing the tensorboard event file

        Returns:
        --------
        (dict(pandas.DataFrame))
            Dictionary with (key, value) pairs corresponding to the name
            and the data of the metric
        """
        event_acc = EventAccumulator(event_dir)
        event_acc.Reload()
        pandas_dict = {tag: data
                       for (tag, data) in cls._extract_scalars(event_acc)}
        return pandas_dict

    @staticmethod
    def _extract_scalars(event_acc):
        """Extract scalar quantities from event accumulator."""
        for tag in event_acc.Tags()['scalars']:
            wall, step, value = zip(*event_acc.Scalars(tag))
            data = pandas.DataFrame(dict(wall=wall, step=step, value=value))
            yield (tag, data)
