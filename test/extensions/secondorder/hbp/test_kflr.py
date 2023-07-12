"""Test BackPACK's KFLR extension."""

from test.automated_test import check_sizes_and_values
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import ExtensionsTestProblem, make_test_problems
from test.extensions.secondorder.hbp.kflr_settings import (
    BATCH_SIZE_1_SETTINGS,
    NOT_SUPPORTED_SETTINGS,
)

import pytest

NOT_SUPPORTED_PROBLEMS = make_test_problems(NOT_SUPPORTED_SETTINGS)
NOT_SUPPORTED_IDS = [problem.make_id() for problem in NOT_SUPPORTED_PROBLEMS]
BATCH_SIZE_1_PROBLEMS = make_test_problems(BATCH_SIZE_1_SETTINGS)
BATCH_SIZE_1_IDS = [problem.make_id() for problem in BATCH_SIZE_1_PROBLEMS]


@pytest.mark.parametrize("problem", NOT_SUPPORTED_PROBLEMS, ids=NOT_SUPPORTED_IDS)
def test_kflr_not_supported(problem):
    """Check that the KFLR extension does not allow specific hyperparameters/modules.

    Args:
        problem (ExtensionsTestProblem): Test case.
    """
    problem.set_up()

    with pytest.raises(RuntimeError):
        BackpackExtensions(problem).kflr()

    problem.tear_down()


@pytest.mark.parametrize("problem", BATCH_SIZE_1_PROBLEMS, ids=BATCH_SIZE_1_IDS)
def test_kflr_equals_ggn(problem: ExtensionsTestProblem):
    """Check that for batch_size = 1 and linear layers, KFLR is the GGN block.

    Args:
        problem: Test case.
    """
    problem.set_up()

    autograd_res = AutogradExtensions(problem).ggn_blocks()
    backpack_res = BackpackExtensions(problem).kflr_as_mat()

    check_sizes_and_values(autograd_res, backpack_res, atol=1e-7, rtol=1e-5)

    problem.tear_down()
