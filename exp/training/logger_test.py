"""Test logging and extraction to .csv."""

import numpy as np
from .logger import Logger
from ..utils import directory_in_data


def test_pandas_conversion():
    """Log a value and convert to pandas DataFrame."""
    logger = Logger(directory_in_data('test_logging'))
    logger.log_value('metric1', 0.5, 1)
    logger.log_value('metric2', 0.3, 5)
    data = logger.extract_logdir_to_pandas()

    df1 = data['metric1'][['step', 'value']].values
    array1 = np.array([1, 0.5])
    assert array1 in df1

    df2 = data['metric2'][['step', 'value']].values
    array2 = np.array([5, 0.3])
    assert array2 in df2
