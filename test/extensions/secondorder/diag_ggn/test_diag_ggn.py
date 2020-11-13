from test.automated_test import check_sizes_and_values
from test.extensions.problem import make_test_problems
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.secondorder.diag_ggn.diaggnn_settings import DiagGGN_SETTINGS

import pytest


PROBLEMS = make_test_problems(DiagGGN_SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_diag_ggn(problem):
    """Test the diagonal of Gauss-Newton

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    problem.set_up()

    backpack_res = BackpackExtensions(problem).diag_ggn()
    autograd_res = AutogradExtensions(problem).diag_ggn()

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_diag_ggn_mc_light(problem):
    """Test the MC approximation of Diagonal of Gauss-Newton
        with few mc_samples (light version)

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    problem.set_up()

    backpack_res = BackpackExtensions(problem).diag_ggn()

    mc_samples = 1000
    backpack_res_mc_avg = BackpackExtensions(problem).diag_ggn_mc(mc_samples)

    check_sizes_and_values(backpack_res, backpack_res_mc_avg, atol=1e-4, rtol=1e-1)
    problem.tear_down()


@pytest.mark.montecarlo
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_diag_ggn_mc(problem):
    """Test the MC approximation of Diagonal of Gauss-Newton
       with more samples (slow version)

    Note: This function runs for 100,000 mc_samples. In case this function crashes,
    alternate approach is to loop 10 times over 10,000 samples & then average them.

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    problem.set_up()

    backpack_res = BackpackExtensions(problem).diag_ggn()

    mc_samples = 100000
    backpack_res_mc_avg = BackpackExtensions(problem).diag_ggn_mc(mc_samples)

    check_sizes_and_values(backpack_res, backpack_res_mc_avg, atol=1e-5, rtol=1e-2)
    problem.tear_down()
