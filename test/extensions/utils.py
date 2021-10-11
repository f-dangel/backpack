"""Utility functions for testing BackPACK's extensions."""

from test.extensions.problem import ExtensionsTestProblem
from typing import List, Union

from pytest import skip


def skip_if_subsampling_conflict(
    problem: ExtensionsTestProblem, subsampling: Union[List[int], None]
) -> None:
    """Skip if some samples in subsampling are not contained in input.

    Args:
        problem: Test case.
        subsampling: Indices of active samples.
    """
    N = problem.input.shape[0]
    enough_samples = subsampling is None or N > max(subsampling)
    if not enough_samples:
        skip(f"Not enough samples: N={N}, subsampling={subsampling}")
