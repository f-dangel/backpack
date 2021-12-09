"""Test BackPACK's KFAC extension."""
from test.automated_test import check_sizes_and_values
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import ExtensionsTestProblem, make_test_problems
from test.extensions.secondorder.hbp.kfac_settings import (
    BATCH_SIZE_1_SETTINGS,
    NOT_SUPPORTED_SETTINGS,
)

import pytest

from backpack.utils.kroneckers import kfacs_to_mat

NOT_SUPPORTED_PROBLEMS = make_test_problems(NOT_SUPPORTED_SETTINGS)
NOT_SUPPORTED_IDS = [problem.make_id() for problem in NOT_SUPPORTED_PROBLEMS]
BATCH_SIZE_1_PROBLEMS = make_test_problems(BATCH_SIZE_1_SETTINGS)
BATCH_SIZE_1_IDS = [problem.make_id() for problem in BATCH_SIZE_1_PROBLEMS]


@pytest.mark.parametrize("problem", NOT_SUPPORTED_PROBLEMS, ids=NOT_SUPPORTED_IDS)
def test_kfac_not_supported(problem):
    """Check that the KFAC extension does not allow specific hyperparameters/modules.

    Args:
        problem (ExtensionsTestProblem): Test case.
    """
    problem.set_up()

    with pytest.raises(NotImplementedError):
        BackpackExtensions(problem).kfac()

    problem.tear_down()


@pytest.mark.montecarlo
@pytest.mark.parametrize("problem", BATCH_SIZE_1_PROBLEMS, ids=BATCH_SIZE_1_IDS)
def test_kfac_should_approx_ggn_montecarlo(problem: ExtensionsTestProblem):
    """Check that for batch_size = 1, the K-FAC is the same as the GGN.

    Should be true for linear layers and in the limit of infinite mc_samples.

    Args:
        problem: Test case.
    """
    problem.set_up()
    autograd_res = AutogradExtensions(problem).ggn_blocks()

    mc_samples = 300000
    backpack_kfac = BackpackExtensions(problem).kfac_chunk(mc_samples)
    backpack_res = [kfacs_to_mat(kfac) for kfac in backpack_kfac]

    check_sizes_and_values(autograd_res, backpack_res, atol=5e-3, rtol=5e-3)

    problem.tear_down()


@pytest.mark.parametrize("problem", BATCH_SIZE_1_PROBLEMS, ids=BATCH_SIZE_1_IDS)
def test_kfac_should_approx_ggn_montecarlo_light(problem: ExtensionsTestProblem):
    """Check that for batch_size = 1, the K-FAC is the same as the GGN.

     Should be true for linear layers and in the limit of infinite mc_samples.

    Args:
        problem: Test case.
    """
    problem.set_up()
    autograd_res = AutogradExtensions(problem).ggn_blocks()

    mc_samples = 6000
    backpack_kfac = BackpackExtensions(problem).kfac(mc_samples)
    backpack_res = [kfacs_to_mat(kfac) for kfac in backpack_kfac]

    check_sizes_and_values(autograd_res, backpack_res, atol=1e-2, rtol=1e-2)

    problem.tear_down()
