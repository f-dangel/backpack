"""Test BackPACK's KFRA extension."""

from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import make_test_problems
from test.extensions.secondorder.hbp.kfra_settings import NOT_SUPPORTED_SETTINGS

import pytest

NOT_SUPPORTED_PROBLEMS = make_test_problems(NOT_SUPPORTED_SETTINGS)
NOT_SUPPORTED_IDS = [problem.make_id() for problem in NOT_SUPPORTED_PROBLEMS]


@pytest.mark.parametrize("problem", NOT_SUPPORTED_PROBLEMS, ids=NOT_SUPPORTED_IDS)
def test_kfra_not_supported(problem):
    """Check that the KFRA extension does not allow specific hyperparameters/modules.

    Args:
        problem (ExtensionsTestProblem): Test case.
    """
    problem.set_up()

    with pytest.raises(NotImplementedError):
        BackpackExtensions(problem).kfra()

    problem.tear_down()
