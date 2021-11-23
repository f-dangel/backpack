"""Test BackPACK's KFAC extension."""
import torch

from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import make_test_problems
from test.extensions.secondorder.hbp.kfac_settings import NOT_SUPPORTED_SETTINGS
from test.automated_test import check_sizes_and_values
from test.extensions.secondorder.hbp.kfac_settings import BATCH_SIZE_1_SETTINGS
from backpack.utils.kroneckers import kfacs_to_mat
from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist

import pytest

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


@pytest.mark.parametrize("problem", BATCH_SIZE_1_PROBLEMS, ids=BATCH_SIZE_1_IDS)
def test_kfac_should_approx_ggn_montecarlo(problem):
    """Check that for batch_size = 1, the K-FAC is the same as the GGN

    Args:
        problem (ExtensionsTestProblem): Test case.
    """
    problem.set_up()
    torch.manual_seed(0)
    # calculate GGN
    # create ID matrix for each layer
    mat_list = []
    for p in problem.model.parameters():
        mat_list.append(
            torch.eye(p.numel(), device=p.device).reshape(p.numel(), *p.shape)
        )
    # get output and loss
    outputs = problem.model(problem.input)
    loss = problem.loss_function(outputs, problem.target)
    autograd_res = []
    for layer, mat in zip(problem.model.parameters(), mat_list):
        ggn_cols = []
        for i in range(mat.size(0)):
            e_d = mat[i, :]
            GGN_col_i = ggn_vector_product_from_plist(loss, outputs, [layer], e_d)[0]
            ggn_cols.append(GGN_col_i.unsqueeze(0))
        autograd_res.append(torch.cat(ggn_cols, dim=0).reshape(layer.numel(), layer.numel()))

    # calculate backpack average
    mc_samples = 200
    backpack_average_res = []
    backpack_kfac = BackpackExtensions(problem).kfac(mc_samples)
    for bpr in backpack_kfac:
        backpack_average_res.append(kfacs_to_mat(bpr))

    # check the values
    check_sizes_and_values(autograd_res, backpack_average_res, atol=1e-1, rtol=1e-1)

    problem.tear_down()
