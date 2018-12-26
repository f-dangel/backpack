"""Test merging of .csv files."""

from os import path, makedirs
import pandas
import numpy as np
from pandas.testing import assert_frame_equal

from .merger import CSVMerger
from ..utils import directory_in_data

# create directory to write test .csv files to
test_dir = directory_in_data('test_csv_merging')
makedirs(test_dir, exist_ok=True)

# example data frames: a consistent with b, inconsistent with c
a = pandas.DataFrame({
                      'step': [1, 2],
                      'wall': [0.1, 0.5],
                      'value': [0.5, 0.4],
                      'bar': [3.14159, 43]
                     })
a_file = path.join(test_dir, 'a.csv')
a.to_csv(a_file)

b = pandas.DataFrame({
                      'step': [1, 2],
                      'wall': [1.1, 1.5],
                      'value': [0.7, 1.2],
                      'random': [13, 99]
                     })
b_file = path.join(test_dir, 'b.csv')
b.to_csv(b_file, index=False)

c = pandas.DataFrame({
                      'step': [1, 5, 10],
                      'wall': [10.6, 11.8, 13],
                      'value': [0.3, 0.1, 0.5],
                      'foo': [1.111, 2.222, 3.333]
                     })
c_file = path.join(test_dir, 'c.csv')
c.to_csv(c_file, index=False)


def test_load_and_rename_cols():
    """Test loading of a dataframe and renaming of `value` column."""
    source_files = [a_file]
    labels = ['renamed_value']
    merger = CSVMerger(source_files, labels)

    renamed = merger._load_and_rename_col(source_files[0], labels[0])
    result = pandas.DataFrame({
                               'step': [1, 2],
                               'renamed_value': [0.5, 0.4]
                              })
    assert np.allclose(renamed.values, result.values)


def test_merge_frames_and_consistency():
    """Test merging of frames.""" 
    source_files = [a_file, b_file]
    labels = ['a_value', 'b_value']
    merged = CSVMerger(source_files, labels).merge()
    result = pandas.DataFrame({
                               'step': [1, 2],
                               'a_value': [0.5, 0.4],
                               'b_value': [0.7, 1.2],
                               'mean': [0.6, 0.8],
                               'std': [np.sqrt(0.02), np.sqrt(0.32)]
                              })
    columns = set({*(list(merged) + list(result))})
    for col in columns:
        assert np.allclose(merged[col].values, result[col].values)


def test_merge_frames_inconsistent():
    """Test whether exception is raised when merging inconsistent frames."""
    source_files = [a_file, c_file]
    labels = ['a_value', 'c_value']
    merger = CSVMerger(source_files, labels)
    try:
        merger.merge()
    except CSVMerger.InconsistentCSVError:
        pass
