"""Test class for module partial derivatives.

- Jacobian-matrix products
- Transposed Jacobian-matrix products

- Jacobian-matrix products with respect to layer parameters
- Transposed Jacobian-matrix products with respect to layer parameters
"""

from test.automated_test import check_sizes_and_values
from test.core.derivatives.implementation.autograd import AutogradDerivatives
from test.core.derivatives.implementation.backpack import BackpackDerivatives
from test.core.derivatives.problem import make_test_problems
from test.core.derivatives.settings import SETTINGS
from test.core.derivatives.loss_settings import LOSS_FAIL_SETTINGS
from test.core.derivatives.convolution_settings import CONVOLUTION_FAIL_SETTINGS

import pytest
import torch

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]

NO_LOSS_PROBLEMS = [problem for problem in PROBLEMS if not problem.is_loss()]
NO_LOSS_IDS = [problem.make_id() for problem in NO_LOSS_PROBLEMS]

LOSS_PROBLEMS = [problem for problem in PROBLEMS if problem.is_loss()]
LOSS_IDS = [problem.make_id() for problem in LOSS_PROBLEMS]

# second-order does not make sense
LOSS_FAIL_PROBLEMS = make_test_problems(LOSS_FAIL_SETTINGS)
LOSS_FAIL_IDS = [problem.make_id() for problem in LOSS_FAIL_PROBLEMS]

CONVOLUTION_FAIL_PROBLEMS = make_test_problems(CONVOLUTION_FAIL_SETTINGS)
CONVOLUTION_FAIL_IDS = [problem.make_id() for problem in CONVOLUTION_FAIL_PROBLEMS]


@pytest.mark.parametrize("problem", NO_LOSS_PROBLEMS, ids=NO_LOSS_IDS)
def test_jac_mat_prod(problem, V=3):
    """Test the Jacobian-matrix product.

    Args:
        problem (DerivativesProblem): Problem for derivative test.
        V (int): Number of vectorized Jacobian-vector products.
    """
    problem.set_up()
    mat = torch.rand(V, *problem.input_shape).to(problem.device)

    backpack_res = BackpackDerivatives(problem).jac_mat_prod(mat)
    autograd_res = AutogradDerivatives(problem).jac_mat_prod(mat)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


@pytest.mark.parametrize("problem", NO_LOSS_PROBLEMS, ids=NO_LOSS_IDS)
def test_jac_t_mat_prod(problem, V=3):
    """Test the transposed Jacobian-matrix product.

    Args:
        problem (DerivativesProblem): Problem for derivative test.
        V (int): Number of vectorized transposed Jacobian-vector products.
    """
    problem.set_up()
    mat = torch.rand(V, *problem.output_shape).to(problem.device)

    backpack_res = BackpackDerivatives(problem).jac_t_mat_prod(mat)
    autograd_res = AutogradDerivatives(problem).jac_t_mat_prod(mat)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


PROBLEMS_WITH_WEIGHTS = []
IDS_WITH_WEIGHTS = []
for problem, problem_id in zip(PROBLEMS, IDS):
    if problem.has_weight():
        PROBLEMS_WITH_WEIGHTS.append(problem)
        IDS_WITH_WEIGHTS.append(problem_id)


@pytest.mark.parametrize(
    "sum_batch", [True, False], ids=["sum_batch=True", "sum_batch=False"]
)
@pytest.mark.parametrize(
    "problem",
    PROBLEMS_WITH_WEIGHTS,
    ids=IDS_WITH_WEIGHTS,
)
def test_weight_jac_t_mat_prod(problem, sum_batch, V=3):
    """Test the transposed Jacobian-matrix product w.r.t. to the weights.

    Args:
        problem (DerivativesProblem): Problem for derivative test.
        sum_batch (bool): Sum results over the batch dimension.
        V (int): Number of vectorized transposed Jacobian-vector products.
    """
    problem.set_up()
    mat = torch.rand(V, *problem.output_shape).to(problem.device)

    backpack_res = BackpackDerivatives(problem).weight_jac_t_mat_prod(mat, sum_batch)
    autograd_res = AutogradDerivatives(problem).weight_jac_t_mat_prod(mat, sum_batch)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


@pytest.mark.parametrize(
    "problem",
    PROBLEMS_WITH_WEIGHTS,
    ids=IDS_WITH_WEIGHTS,
)
def test_weight_jac_mat_prod(problem, V=3):
    """Test the Jacobian-matrix product w.r.t. to the weights.

    Args:
        problem (DerivativesProblem): Problem for derivative test.
        V (int): Number of vectorized transposed Jacobian-vector products.
    """
    problem.set_up()
    mat = torch.rand(V, *problem.module.weight.shape).to(problem.device)

    backpack_res = BackpackDerivatives(problem).weight_jac_mat_prod(mat)
    autograd_res = AutogradDerivatives(problem).weight_jac_mat_prod(mat)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


PROBLEMS_WITH_BIAS = []
IDS_WITH_BIAS = []
for problem, problem_id in zip(PROBLEMS, IDS):
    if problem.has_bias():
        PROBLEMS_WITH_BIAS.append(problem)
        IDS_WITH_BIAS.append(problem_id)


@pytest.mark.parametrize(
    "sum_batch", [True, False], ids=["sum_batch=True", "sum_batch=False"]
)
@pytest.mark.parametrize(
    "problem",
    PROBLEMS_WITH_BIAS,
    ids=IDS_WITH_BIAS,
)
def test_bias_jac_t_mat_prod(problem, sum_batch, V=3):
    """Test the transposed Jacobian-matrix product w.r.t. to the biass.

    Args:
        problem (DerivativesProblem): Problem for derivative test.
        sum_batch (bool): Sum results over the batch dimension.
        V (int): Number of vectorized transposed Jacobian-vector products.
    """
    problem.set_up()
    mat = torch.rand(V, *problem.output_shape).to(problem.device)

    backpack_res = BackpackDerivatives(problem).bias_jac_t_mat_prod(mat, sum_batch)
    autograd_res = AutogradDerivatives(problem).bias_jac_t_mat_prod(mat, sum_batch)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


@pytest.mark.parametrize(
    "problem",
    PROBLEMS_WITH_BIAS,
    ids=IDS_WITH_BIAS,
)
def test_bias_jac_mat_prod(problem, V=3):
    """Test the Jacobian-matrix product w.r.t. to the biass.

    Args:
        problem (DerivativesProblem): Problem for derivative test.
        V (int): Number of vectorized transposed Jacobian-vector products.
    """
    problem.set_up()
    mat = torch.rand(V, *problem.module.bias.shape).to(problem.device)

    backpack_res = BackpackDerivatives(problem).bias_jac_mat_prod(mat)
    autograd_res = AutogradDerivatives(problem).bias_jac_mat_prod(mat)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


@pytest.mark.parametrize("problem", LOSS_PROBLEMS, ids=LOSS_IDS)
def test_sqrt_hessian_squared_equals_hessian(problem):
    """Test the sqrt decomposition of the input Hessian.

    Args:
        problem (DerivativesProblem): Problem for derivative test.

    Compares the Hessian to reconstruction from individual Hessian sqrt.
    """
    problem.set_up()

    backpack_res = BackpackDerivatives(problem).input_hessian_via_sqrt_hessian()
    autograd_res = AutogradDerivatives(problem).input_hessian()

    print(backpack_res.device)
    print(autograd_res.device)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


@pytest.mark.parametrize("problem", CONVOLUTION_FAIL_PROBLEMS, ids=CONVOLUTION_FAIL_IDS)
def test_weight_jac_mat_prod_should_fail(problem):
    with pytest.raises(NotImplementedError):
        test_weight_jac_mat_prod(problem)


@pytest.mark.parametrize(
    "sum_batch", [True, False], ids=["sum_batch=True", "sum_batch=False"]
)
@pytest.mark.parametrize("problem", CONVOLUTION_FAIL_PROBLEMS, ids=CONVOLUTION_FAIL_IDS)
def test_weight_jac_t_mat_prod_should_fail(problem, sum_batch):
    with pytest.raises(NotImplementedError):
        test_weight_jac_t_mat_prod(problem, sum_batch)


@pytest.mark.parametrize("problem", LOSS_FAIL_PROBLEMS, ids=LOSS_FAIL_IDS)
def test_sqrt_hessian_should_fail(problem):
    with pytest.raises(ValueError):
        test_sqrt_hessian_squared_equals_hessian(problem)


@pytest.mark.parametrize("problem", LOSS_PROBLEMS, ids=LOSS_IDS)
def test_sqrt_hessian_sampled_squared_approximates_hessian(problem, mc_samples=100000):
    """Test the MC-sampled sqrt decomposition of the input Hessian.

    Args:
        problem (DerivativesProblem): Problem for derivative test.

    Compares the Hessian to reconstruction from individual Hessian MC-sampled sqrt.
    """
    problem.set_up()

    backpack_res = BackpackDerivatives(problem).input_hessian_via_sqrt_hessian(
        mc_samples=mc_samples
    )
    autograd_res = AutogradDerivatives(problem).input_hessian()

    RTOL, ATOL = 1e-2, 2e-2
    check_sizes_and_values(autograd_res, backpack_res, rtol=RTOL, atol=ATOL)
    problem.tear_down()


@pytest.mark.parametrize("problem", LOSS_FAIL_PROBLEMS, ids=LOSS_FAIL_IDS)
def test_sqrt_hessian_sampled_should_fail(problem):
    with pytest.raises(ValueError):
        test_sqrt_hessian_sampled_squared_approximates_hessian(problem)


@pytest.mark.parametrize("problem", LOSS_PROBLEMS, ids=LOSS_IDS)
def test_sum_hessian(problem):
    """Test the summed Hessian.

    Args:
        problem (DerivativesProblem): Problem for derivative test.
    """
    problem.set_up()

    backpack_res = BackpackDerivatives(problem).sum_hessian()
    autograd_res = AutogradDerivatives(problem).sum_hessian()

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


@pytest.mark.parametrize("problem", LOSS_FAIL_PROBLEMS, ids=LOSS_FAIL_IDS)
def test_sum_hessian_should_fail(problem):
    with pytest.raises(ValueError):
        test_sum_hessian(problem)


@pytest.mark.parametrize("problem", NO_LOSS_PROBLEMS, ids=NO_LOSS_IDS)
def test_ea_jac_t_mat_jac_prod(problem):
    """Test KFRA backpropagation

    H_in →  1/N ∑ₙ Jₙ^T H_out Jₙ

    Notes:
        - `Dropout` cannot be tested,as the `autograd` implementation does a forward
        pass over each sample, while the `backpack` implementation requires only
        one forward pass over the batched data. This leads to different outputs,
        as `Dropout` is not deterministic.

    Args:
        problem (DerivativesProblem): Problem for derivative test.
    """
    problem.set_up()
    out_features = torch.prod(torch.tensor(problem.output_shape[1:]))
    mat = torch.rand(out_features, out_features).to(problem.device)

    backpack_res = BackpackDerivatives(problem).ea_jac_t_mat_jac_prod(mat)
    autograd_res = AutogradDerivatives(problem).ea_jac_t_mat_jac_prod(mat)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


@pytest.mark.skip("[WAITING] Autograd issue with Hessian-vector products")
@pytest.mark.parametrize("problem", NO_LOSS_PROBLEMS, ids=NO_LOSS_IDS)
def test_hessian_is_zero(problem):
    """Check if the input-output Hessian is (non-)zero."""
    problem.set_up()

    backpack_res = BackpackDerivatives(problem).hessian_is_zero()
    autograd_res = AutogradDerivatives(problem).hessian_is_zero()

    assert backpack_res == autograd_res
    problem.tear_down()


@pytest.mark.skip
@pytest.mark.parametrize("problem", NO_LOSS_PROBLEMS, ids=NO_LOSS_IDS)
def test_hessian_is_diagonal(problem):
    problem.set_up()

    # TODO
    raise NotImplementedError

    problem.tear_down()


@pytest.mark.skip
@pytest.mark.parametrize("problem", NO_LOSS_PROBLEMS, ids=NO_LOSS_IDS)
def test_hessian_diagonal(problem):
    problem.set_up()

    # TODO
    raise NotImplementedError

    problem.tear_down()


@pytest.mark.skip
@pytest.mark.parametrize("problem", NO_LOSS_PROBLEMS, ids=NO_LOSS_IDS)
def test_hessian_is_psd(problem):
    problem.set_up()

    # TODO
    raise NotImplementedError

    problem.tear_down()
