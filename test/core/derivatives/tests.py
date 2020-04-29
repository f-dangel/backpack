"""Test class for module partial derivatives.

- Jacobian-matrix products
- Transposed Jacobian-matrix products
"""

from test.automated_test import check_sizes, check_values
from test.core.derivatives.convert import make_test_problems
from test.core.derivatives.implementation.autograd import AutogradDerivatives
from test.core.derivatives.implementation.backpack import BackpackDerivatives
from test.core.derivatives.settings import SETTINGS

import pytest
import torch

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_jac_mat_prod(problem, V=3):
    """Test the Jacobian-matrix product.

    Args:
        problem (DerivativesProblem): Problem for derivative test.
        V (int): Number of vectorized Jacobian-vector products.
    """
    torch.manual_seed(123)
    mat = torch.rand(V, *problem.in_shape).to(problem.device)

    backpack_res = BackpackDerivatives(problem).jac_mat_prod(mat)
    autograd_res = AutogradDerivatives(problem).jac_mat_prod(mat)

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_jac_t_mat_prod(problem, V=3):
    """Test the transposed Jacobian-matrix product.

    Args:
        problem (DerivativesProblem): Problem for derivative test.
        V (int): Number of vectorized transposed Jacobian-vector products.
    """
    torch.manual_seed(123)
    mat = torch.rand(V, *problem.out_shape).to(problem.device)

    backpack_res = BackpackDerivatives(problem).jac_t_mat_prod(mat)
    autograd_res = AutogradDerivatives(problem).jac_t_mat_prod(mat)

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)
