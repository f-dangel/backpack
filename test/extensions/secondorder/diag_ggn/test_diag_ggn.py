"""Test DiagGGN extension."""
from test.automated_test import check_sizes_and_values
from test.core.derivatives.utils import regression_targets
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import make_test_problems
from test.extensions.secondorder.diag_ggn.diag_ggn_settings import DiagGGN_SETTINGS

import pytest
import torch
from torch.nn import RNN, Flatten, Sequential

from backpack.custom_module.permute import Permute
from backpack.custom_module.reduce_tuple import ReduceTuple

LOCAL_SETTINGS = [
    # RNN settings
    {
        "input_fn": lambda: torch.rand(8, 5, 6),
        "module_fn": lambda: Sequential(
            Permute(dims=[1, 0, 2]),
            RNN(input_size=6, hidden_size=3),
            ReduceTuple(index=0),
            Permute(dims=[1, 2, 0]),
            Flatten(),
        ),
        "loss_function_fn": lambda: torch.nn.MSELoss(),
        "target_fn": lambda: regression_targets((8, 3 * 5)),
    },
]

PROBLEMS = make_test_problems(DiagGGN_SETTINGS + LOCAL_SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_diag_ggn(problem, request):
    """Test the diagonal of Gauss-Newton.

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
        request: contains information about test
    """
    problem.set_up()

    backpack_res = BackpackExtensions(problem).diag_ggn()
    if all(string in request.node.callspec.id for string in ["RNN", "cuda"]):
        # torch does not implement cuda double-backwards pass on RNNs and
        # recommends this workaround
        with torch.backends.cudnn.flags(enabled=False):
            autograd_res = AutogradExtensions(problem).diag_ggn()
    else:
        autograd_res = AutogradExtensions(problem).diag_ggn()

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


MC_ATOL = 1e-4
MC_LIGHT_RTOL = 1e-1
MC_RTOL = 1e-2


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_diag_ggn_mc_light(problem):
    """Test the MC approximation of Diagonal of Gauss-Newton.

    with few mc_samples (light version)

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    problem.set_up()

    backpack_res = BackpackExtensions(problem).diag_ggn()
    mc_samples = 3000
    backpack_res_mc_avg = BackpackExtensions(problem).diag_ggn_mc(mc_samples)

    check_sizes_and_values(
        backpack_res, backpack_res_mc_avg, atol=MC_ATOL, rtol=MC_LIGHT_RTOL
    )
    problem.tear_down()


@pytest.mark.montecarlo
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_diag_ggn_mc(problem):
    """Test the MC approximation of Diagonal of Gauss-Newton.

    with more samples (slow version)

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    problem.set_up()

    backpack_res = BackpackExtensions(problem).diag_ggn()
    mc_samples = 300000
    chunks = 30
    backpack_res_mc_avg = BackpackExtensions(problem).diag_ggn_mc_chunk(
        mc_samples, chunks=chunks
    )

    check_sizes_and_values(
        backpack_res, backpack_res_mc_avg, atol=MC_ATOL, rtol=MC_RTOL
    )
    problem.tear_down()
