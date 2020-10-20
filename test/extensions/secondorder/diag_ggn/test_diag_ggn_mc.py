from test.automated_test import check_sizes_and_values
from test.extensions.problem import make_test_problems
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.secondorder.diag_ggn.diaggnn_settings import DiagGGN_SETTINGS

import pytest


PROBLEMS = make_test_problems(DiagGGN_SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_batch_grad(problem):
    """Test individual gradients

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    problem.set_up()

    backpack_res = BackpackExtensions(problem).diag_ggn_mc()
    autograd_res = AutogradExtensions(problem).diag_ggn()

    check_sizes_and_values(autograd_res, backpack_res, atol=1e-1, rtol=1e-1)
    problem.tear_down()
