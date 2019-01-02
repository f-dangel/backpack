"""Merge .csv files from logging with tensorboard.

Assumes special structure of the .csv files.
"""

import pandas


class CSVMerger():
    """Handle merging of .csv files extracted from tensorboard logging."""
    def __init__(self, source_files, labels):
        """
        Parameters:
        -----------
        source_files : (list(str))
            List of source .csv files extracted from tensorboard
        labels : (list(str))
            New labels of the `'value'` columns for each file in
            `source_files`
        """
        self.source_files = source_files
        self.labels = labels

    def merge(self):
        """Return merged dataframe with mean and stddev column."""
        data = [self._load_and_rename_col(file, label)
                for file, label in zip(self.source_files, self.labels)]
        df = self._merge_frames_and_check_consistency(data)
        df = self._add_mean_col(df, self.labels)
        df = self._add_std_col(df, self.labels)
        return df

    def _load_and_rename_col(self, source_file, new_label):
        """Load file as pandas.DataFrame, rename `'value'` column.
        
        Wall time is ignored.
        """
        rename_dict = {
                       'step': 'step', 
                       'value': new_label
                       }
        df = pandas.read_csv(source_file)[list(rename_dict.keys())]
        return df.rename(index=str, columns=rename_dict)

    def _merge_frames_and_check_consistency(self, data_frames):
        """Merge data frames on column `'step'`. Check for NaNs.

        NaNs occur if the source data frames were not logged at
        the same step.

        Raises:
        -------
        (InconsistentCSVError)
            If NaNs emerge in the merged version
        """
        temp = None
        while data_frames:
            df = data_frames.pop()
            temp = df if temp is None else temp.merge(df, on=['step'])
        self._raise_exception_if_nan(temp)
        return temp

    class InconsistentCSVError(Exception):
        """Exception raised if CSV data is inconsistent."""
        pass

    def _raise_exception_if_nan(self, df):
        """Raise inconsistency exception if data frame contains NaN."""
        num_nans = df.isnull().sum().sum()
        if num_nans != 0:
            raise self.InconsistentCSVError('Found {} NaN values'
                                            .format(num_nans))

    @staticmethod
    def _add_mean_col(df, labels):
        """Compute mean of columns given by `labels` in `df`.

        Add column `'mean'` to data frame.
        """
        df['mean'] = df[labels].mean(axis=1)
        return df

    @staticmethod
    def _add_std_col(df, labels):
        """Compute stddev of columns given by `labels` in `df`.

        Add column `'std'` to data frame.
        """
        df['std'] = df[labels].std(axis=1)
        return df
