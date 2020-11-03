from test.automated_test import check_sizes_and_values
from test.extensions.problem import make_test_problems
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.secondorder.diag_hessian.diagh_settings import DiagHESSIAN_SETTINGS

import pytest


PROBLEMS = make_test_problems(DiagHESSIAN_SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_diag_h(problem):
    """Test Diagonal of Hessian

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    problem.set_up()

    backpack_res = BackpackExtensions(problem).diag_h()
    autograd_res = AutogradExtensions(problem).diag_h()

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()
