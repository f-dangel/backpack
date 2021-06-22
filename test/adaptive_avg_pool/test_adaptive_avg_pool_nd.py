"""Test the shape checker of AdaptiveAvgPoolNDDerivatives."""
from test.adaptive_avg_pool.problem import AdaptiveAvgPoolProblem, make_test_problems
from test.adaptive_avg_pool.settings_adaptive_avg_pool_nd import SETTINGS
from typing import List

import pytest

PROBLEMS: List[AdaptiveAvgPoolProblem] = make_test_problems(SETTINGS)
IDS: List[str] = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_adaptive_avg_pool_check_parameters(problem: AdaptiveAvgPoolProblem):
    """Test AdaptiveAvgPoolNDDerivatives.check_parameters().

    Additionally check if returned parameters are indeed equivalent.

    Args:
        problem: test problem
    """
    problem.set_up()
    if problem.works:
        problem.check_parameters()
        problem.check_equivalence()
    else:
        with pytest.raises(NotImplementedError):
            problem.check_parameters()
    problem.tear_down()
