"""Test BackPACK's KFRA extension."""

from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import ExtensionsTestProblem, make_test_problems
from test.extensions.secondorder.hbp.kfra_settings import (
    BATCH_SIZE_1_SETTINGS,
    NOT_SUPPORTED_SETTINGS,
)
from test.utils.skip_extension_test import skip_BCEWithLogitsLoss

import pytest
from torch import Tensor, prod

NOT_SUPPORTED_PROBLEMS = make_test_problems(NOT_SUPPORTED_SETTINGS)
NOT_SUPPORTED_IDS = [problem.make_id() for problem in NOT_SUPPORTED_PROBLEMS]
BATCH_SIZE_1_PROBLEMS = make_test_problems(BATCH_SIZE_1_SETTINGS)
BATCH_SIZE_1_IDS = [problem.make_id() for problem in BATCH_SIZE_1_PROBLEMS]


@pytest.mark.parametrize("problem", NOT_SUPPORTED_PROBLEMS, ids=NOT_SUPPORTED_IDS)
def test_kfra_not_supported(problem: ExtensionsTestProblem):
    """Check that the KFRA extension does not allow specific hyperparameters/modules.

    Args:
        problem: Test case.
    """
    problem.set_up()

    with pytest.raises(NotImplementedError):
        BackpackExtensions(problem).kfra()

    problem.tear_down()


@pytest.mark.parametrize("problem", BATCH_SIZE_1_PROBLEMS, ids=BATCH_SIZE_1_IDS)
def test_kfra_dimensions(problem: ExtensionsTestProblem):
    """Check that block Hessian approximation of KFRA has correct dimension.

    This test runs KFRA code, but due to the approximations made in KFRA, a case
    where it becomes exact and can therefore be tested for correct values still
    needs to be identified.

    Args:
        problem: Test case.
    """
    problem.set_up()
    skip_BCEWithLogitsLoss(problem)

    backpack_kfra = BackpackExtensions(problem).kfra()
    for p, p_kfra in zip(problem.trainable_parameters(), backpack_kfra):
        assert all(kron.dim() == 2 for kron in p_kfra)
        assert all(kron.shape[0] == kron.shape[1] for kron in p_kfra)

        kron_dims = Tensor([kron_fac.shape[0] for kron_fac in p_kfra])
        assert p.numel() == prod(kron_dims)
