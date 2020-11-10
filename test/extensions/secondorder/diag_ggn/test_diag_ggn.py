from test.automated_test import check_sizes_and_values
from test.extensions.problem import make_test_problems
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.secondorder.diag_ggn.diaggnn_settings import DiagGGN_SETTINGS

import pytest
import torch


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

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    problem.set_up()
    torch.manual_seed(0)

    backpack_res = BackpackExtensions(problem).diag_ggn()
    backpack_res_mc_avg = []
    for param_res in backpack_res:
        backpack_res_mc_avg.append(torch.zeros_like(param_res))

    mc_samples = 10
    for _ in range(mc_samples):
        backpack_diagggn_mc = BackpackExtensions(problem).diag_ggn_mc(mc_samples)
        for i, param_res in enumerate(backpack_diagggn_mc):
            backpack_res_mc_avg[i] += param_res

    for i in range(len(backpack_res_mc_avg)):
        backpack_res_mc_avg[i] /= mc_samples

    check_sizes_and_values(backpack_res, backpack_res_mc_avg, atol=1e-1, rtol=1e-1)
    problem.tear_down()


@pytest.mark.slow
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_diag_ggn_mc(problem):
    """Test the MC approximation of Diagonal of Gauss-Newton

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    problem.set_up()
    torch.manual_seed(0)

    backpack_res = BackpackExtensions(problem).diag_ggn()
    backpack_res_mc_avg = []
    for param_res in backpack_res:
        backpack_res_mc_avg.append(torch.zeros_like(param_res))

    mc_samples = 500
    for _ in range(mc_samples):
        backpack_diagggn_mc = BackpackExtensions(problem).diag_ggn_mc(mc_samples)
        for i, param_res in enumerate(backpack_diagggn_mc):
            backpack_res_mc_avg[i] += param_res

    for i in range(len(backpack_res_mc_avg)):
        backpack_res_mc_avg[i] /= mc_samples

    check_sizes_and_values(backpack_res, backpack_res_mc_avg, atol=1e-1, rtol=1e-1)
    problem.tear_down()
