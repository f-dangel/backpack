"""Test class for module partial derivatives.

- Jacobian-matrix products
- Transposed Jacobian-matrix products

- Jacobian-matrix products with respect to layer parameters
- Transposed Jacobian-matrix products with respect to layer parameters
"""
from contextlib import nullcontext
from test.automated_test import check_sizes_and_values
from test.core.derivatives.batch_norm_settings import BATCH_NORM_SETTINGS
from test.core.derivatives.embedding_settings import EMBEDDING_SETTINGS
from test.core.derivatives.implementation.autograd import AutogradDerivatives
from test.core.derivatives.implementation.backpack import BackpackDerivatives
from test.core.derivatives.loss_settings import LOSS_FAIL_SETTINGS
from test.core.derivatives.lstm_settings import LSTM_SETTINGS
from test.core.derivatives.padding_settings import CUSTOM_PADDING_SETTINGS
from test.core.derivatives.permute_settings import PERMUTE_SETTINGS
from test.core.derivatives.problem import DerivativesTestProblem, make_test_problems
from test.core.derivatives.rnn_settings import RNN_SETTINGS as RNN_SETTINGS
from test.core.derivatives.scale_module_settings import SCALE_MODULE_SETTINGS
from test.core.derivatives.settings import SETTINGS
from test.core.derivatives.slicing_settings import CUSTOM_SLICING_SETTINGS
from test.utils.skip_test import (
    skip_adaptive_avg_pool3d_cuda,
    skip_batch_norm_train_mode_with_subsampling,
    skip_subsampling_conflict,
)
from typing import List, Union
from warnings import warn

from pytest import fixture, mark, raises, skip
from torch import Tensor, rand

from backpack.core.derivatives.convnd import weight_jac_t_save_memory

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]

NO_LOSS_PROBLEMS = [problem for problem in PROBLEMS if not problem.is_loss()]
NO_LOSS_IDS = [problem.make_id() for problem in NO_LOSS_PROBLEMS]

LOSS_PROBLEMS = [problem for problem in PROBLEMS if problem.is_loss()]
LOSS_IDS = [problem.make_id() for problem in LOSS_PROBLEMS]

# second-order does not make sense
LOSS_FAIL_PROBLEMS = make_test_problems(LOSS_FAIL_SETTINGS)
LOSS_FAIL_IDS = [problem.make_id() for problem in LOSS_FAIL_PROBLEMS]

RNN_PROBLEMS = make_test_problems(RNN_SETTINGS)
RNN_PROBLEMS += make_test_problems(LSTM_SETTINGS)
RNN_IDS = [problem.make_id() for problem in RNN_PROBLEMS]

PERMUTE_PROBLEMS = make_test_problems(PERMUTE_SETTINGS)
PERMUTE_IDS = [problem.make_id() for problem in PERMUTE_PROBLEMS]

BATCH_NORM_PROBLEMS = make_test_problems(BATCH_NORM_SETTINGS)
BATCH_NORM_IDS = [problem.make_id() for problem in BATCH_NORM_PROBLEMS]

EMBEDDING_PROBLEMS = make_test_problems(EMBEDDING_SETTINGS)
EMBEDDING_IDS = [problem.make_id() for problem in EMBEDDING_PROBLEMS]

SCALE_MODULE_PROBLEMS = make_test_problems(SCALE_MODULE_SETTINGS)
SCALE_MODULE_IDS = [problem.make_id() for problem in SCALE_MODULE_PROBLEMS]

CUSTOM_PADDING_MODULE_PROBLEMS = make_test_problems(CUSTOM_PADDING_SETTINGS)
CUSTOM_PADDING_MODULE_IDS = [
    problem.make_id() for problem in CUSTOM_PADDING_MODULE_PROBLEMS
]

CUSTOM_SLICING_MODULE_PROBLEMS = make_test_problems(CUSTOM_SLICING_SETTINGS)
CUSTOM_SLICING_MODULE_IDS = [
    problem.make_id() for problem in CUSTOM_SLICING_MODULE_PROBLEMS
]

SUBSAMPLINGS = [None, [0, 0], [2, 0]]
SUBSAMPLING_IDS = [f"subsampling={s}".replace(" ", "") for s in SUBSAMPLINGS]


@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLING_IDS)
@mark.parametrize("sum_batch", [True, False], ids=["sum_batch=True", "sum_batch=False"])
def test_param_mjp(
    problem: DerivativesTestProblem,
    sum_batch: bool,
    subsampling: List[int] or None,
    request,
) -> None:
    """Test all parameter derivatives.

    Args:
        problem: test problem
        sum_batch: whether to sum along batch axis
        subsampling: subsampling indices
        request: problem request
    """
    skip_subsampling_conflict(problem, subsampling)
    test_save_memory: bool = "Conv" in request.node.callspec.id
    V = 3

    for param_str, _ in problem.module.named_parameters():
        print(f"testing derivative wrt {param_str}")
        for save_memory in [True, False] if test_save_memory else [None]:
            if test_save_memory:
                print(f"testing with save_memory={save_memory}")

            mat = rand_mat_like_output(V, problem, subsampling=subsampling)
            with weight_jac_t_save_memory(
                save_memory=save_memory
            ) if test_save_memory else nullcontext():
                backpack_res = BackpackDerivatives(problem).param_mjp(
                    param_str, mat, sum_batch, subsampling=subsampling
                )
            autograd_res = AutogradDerivatives(problem).param_mjp(
                param_str, mat, sum_batch, subsampling=subsampling
            )

            check_sizes_and_values(autograd_res, backpack_res)


@mark.parametrize(
    "problem",
    NO_LOSS_PROBLEMS + RNN_PROBLEMS + PERMUTE_PROBLEMS + BATCH_NORM_PROBLEMS,
    ids=NO_LOSS_IDS + RNN_IDS + PERMUTE_IDS + BATCH_NORM_IDS,
)
def test_jac_mat_prod(problem: DerivativesTestProblem, V: int = 3) -> None:
    """Test the Jacobian-matrix product.

    Args:
        problem: Test case.
        V: Number of vectorized Jacobian-vector products. Default: ``3``.
    """
    problem.set_up()
    mat = rand(V, *problem.input_shape).to(problem.device)

    backpack_res = BackpackDerivatives(problem).jac_mat_prod(mat)
    autograd_res = AutogradDerivatives(problem).jac_mat_prod(mat)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLING_IDS)
@mark.parametrize(
    "problem",
    NO_LOSS_PROBLEMS
    + RNN_PROBLEMS
    + PERMUTE_PROBLEMS
    + BATCH_NORM_PROBLEMS
    + SCALE_MODULE_PROBLEMS
    + CUSTOM_PADDING_MODULE_PROBLEMS
    + CUSTOM_SLICING_MODULE_PROBLEMS,
    ids=NO_LOSS_IDS
    + RNN_IDS
    + PERMUTE_IDS
    + BATCH_NORM_IDS
    + SCALE_MODULE_IDS
    + CUSTOM_PADDING_MODULE_IDS
    + CUSTOM_SLICING_MODULE_IDS,
)
def test_jac_t_mat_prod(
    problem: DerivativesTestProblem,
    subsampling: Union[None, List[int]],
    request,
    V: int = 3,
) -> None:
    """Test the transposed Jacobian-matrix product.

    Args:
        problem: Problem for derivative test.
        subsampling: Indices of active samples.
        request: Pytest request, used for getting id.
        V: Number of vectorized transposed Jacobian-vector products. Default: ``3``.
    """
    skip_adaptive_avg_pool3d_cuda(request)

    problem.set_up()
    skip_batch_norm_train_mode_with_subsampling(problem, subsampling)
    skip_subsampling_conflict(problem, subsampling)
    mat = rand_mat_like_output(V, problem, subsampling=subsampling)

    backpack_res = BackpackDerivatives(problem).jac_t_mat_prod(
        mat, subsampling=subsampling
    )
    autograd_res = AutogradDerivatives(problem).jac_t_mat_prod(
        mat, subsampling=subsampling
    )

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


PROBLEMS_WITH_WEIGHTS = []
IDS_WITH_WEIGHTS = []
for problem, problem_id in zip(PROBLEMS, IDS):
    if problem.has_weight():
        PROBLEMS_WITH_WEIGHTS.append(problem)
        IDS_WITH_WEIGHTS.append(problem_id)


def rand_mat_like_output(
    V: int, problem: DerivativesTestProblem, subsampling: List[int] = None
) -> Tensor:
    """Generate random matrix whose columns are shaped like the layer output.

    Can be used to generate random inputs to functions that act on tensors
    shaped like the module output (like ``*_jac_t_mat_prod``).

    Args:
        V: Number of rows.
        problem: Test case.
        subsampling: Indices of samples used by sub-sampling.

    Returns:
        Random matrix with (subsampled) output shape.
    """
    subsample_shape = list(problem.output_shape)

    if subsampling is not None:
        subsample_shape[0] = len(subsampling)

    return rand(V, *subsample_shape, device=problem.device)


@mark.parametrize(
    "problem",
    PROBLEMS_WITH_WEIGHTS + BATCH_NORM_PROBLEMS,
    ids=IDS_WITH_WEIGHTS + BATCH_NORM_IDS,
)
def test_weight_jac_mat_prod(problem: DerivativesTestProblem, V: int = 3) -> None:
    """Test the Jacobian-matrix product w.r.t. to the weight.

    Args:
        problem: Test case.
        V: Number of vectorized Jacobian-vector products. Default: ``3``.
    """
    problem.set_up()
    mat = rand(V, *problem.module.weight.shape).to(problem.device)

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


@mark.parametrize(
    "problem",
    PROBLEMS_WITH_BIAS + BATCH_NORM_PROBLEMS,
    ids=IDS_WITH_BIAS + BATCH_NORM_IDS,
)
def test_bias_jac_mat_prod(problem: DerivativesTestProblem, V: int = 3) -> None:
    """Test the Jacobian-matrix product w.r.t. to the bias.

    Args:
        problem: Test case.
        V: Number of vectorized Jacobian-vector products. Default: ``3``.
    """
    problem.set_up()
    mat = rand(V, *problem.module.bias.shape).to(problem.device)

    backpack_res = BackpackDerivatives(problem).bias_jac_mat_prod(mat)
    autograd_res = AutogradDerivatives(problem).bias_jac_mat_prod(mat)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLING_IDS)
@mark.parametrize("problem", LOSS_PROBLEMS, ids=LOSS_IDS)
def test_sqrt_hessian_squared_equals_hessian(
    problem: DerivativesTestProblem, subsampling: Union[List[int], None]
) -> None:
    """Test the sqrt decomposition of the input Hessian.

    Args:
        problem: Test case.
        subsampling: Indices of active samples.

    Compares the Hessian to reconstruction from individual Hessian sqrt.
    """
    problem.set_up()
    skip_subsampling_conflict(problem, subsampling)

    backpack_res = BackpackDerivatives(problem).input_hessian_via_sqrt_hessian(
        subsampling=subsampling
    )
    autograd_res = AutogradDerivatives(problem).input_hessian(subsampling=subsampling)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLING_IDS)
@mark.parametrize("problem", LOSS_FAIL_PROBLEMS, ids=LOSS_FAIL_IDS)
def test_sqrt_hessian_should_fail(
    problem: DerivativesTestProblem, subsampling: Union[List[int], None]
) -> None:
    """Test that sqrt_hessian fails.

    Args:
        problem: Test case.
        subsampling: Indices of active samples.
    """
    with raises(ValueError):
        test_sqrt_hessian_squared_equals_hessian(problem, subsampling)


@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLING_IDS)
@mark.parametrize("problem", LOSS_PROBLEMS, ids=LOSS_IDS)
def test_sqrt_hessian_sampled_squared_approximates_hessian(
    problem: DerivativesTestProblem,
    subsampling: Union[List[int], None],
    mc_samples: int = 1000000,
    chunks: int = 10,
) -> None:
    """Test the MC-sampled sqrt decomposition of the input Hessian.

    Compares the Hessian to reconstruction from individual Hessian MC-sampled sqrt.

    Args:
        problem: Test case.
        subsampling: Indices of active samples.
        mc_samples: number of samples. Defaults to 1000000.
        chunks: Number of passes the MC samples will be processed sequentially.
    """
    problem.set_up()
    skip_subsampling_conflict(problem, subsampling)

    backpack_res = BackpackDerivatives(problem).input_hessian_via_sqrt_hessian(
        mc_samples=mc_samples, chunks=chunks, subsampling=subsampling
    )
    autograd_res = AutogradDerivatives(problem).input_hessian(subsampling=subsampling)

    RTOL, ATOL = 1e-2, 7e-3
    check_sizes_and_values(autograd_res, backpack_res, rtol=RTOL, atol=ATOL)
    problem.tear_down()


@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLING_IDS)
@mark.parametrize("problem", LOSS_FAIL_PROBLEMS, ids=LOSS_FAIL_IDS)
def test_sqrt_hessian_sampled_should_fail(
    problem: DerivativesTestProblem, subsampling: Union[List[int], None]
) -> None:
    """Test that sqrt_hessian_samples fails.

    Args:
        problem: Test case.
        subsampling: Indices of active samples.
    """
    with raises(ValueError):
        test_sqrt_hessian_sampled_squared_approximates_hessian(problem, subsampling)


@mark.parametrize("problem", LOSS_PROBLEMS, ids=LOSS_IDS)
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


@mark.parametrize("problem", LOSS_FAIL_PROBLEMS, ids=LOSS_FAIL_IDS)
def test_sum_hessian_should_fail(problem):
    """Test sum_hessian, should fail.

    Args:
        problem: test problem
    """
    with raises(ValueError):
        test_sum_hessian(problem)


@mark.parametrize("problem", NO_LOSS_PROBLEMS, ids=NO_LOSS_IDS)
def test_ea_jac_t_mat_jac_prod(problem: DerivativesTestProblem, request) -> None:
    """Test KFRA backpropagation.

    H_in →  1/N ∑ₙ Jₙ^T H_out Jₙ

    Notes:
        - `Dropout` cannot be tested,as the `autograd` implementation does a forward
        pass over each sample, while the `backpack` implementation requires only
        one forward pass over the batched data. This leads to different outputs,
        as `Dropout` is not deterministic.

    Args:
        problem: Test case.
        request: PyTest request, used to get test id.
    """
    skip_adaptive_avg_pool3d_cuda(request)

    problem.set_up()
    out_features = problem.output_shape[1:].numel()
    mat = rand(out_features, out_features).to(problem.device)

    backpack_res = BackpackDerivatives(problem).ea_jac_t_mat_jac_prod(mat)
    autograd_res = AutogradDerivatives(problem).ea_jac_t_mat_jac_prod(mat)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


@fixture(
    params=PROBLEMS
    + BATCH_NORM_PROBLEMS
    + RNN_PROBLEMS
    + EMBEDDING_PROBLEMS
    + CUSTOM_PADDING_MODULE_PROBLEMS
    + CUSTOM_SLICING_MODULE_PROBLEMS,
    ids=lambda p: p.make_id(),
)
def problem(request) -> DerivativesTestProblem:
    """Set seed, create tested layer and data. Finally clean up.

    Args:
        request (SubRequest): Request for the fixture from a test/fixture function.

    Yields:
        Test case with deterministically constructed attributes.
    """
    case = request.param
    case.set_up()
    yield case
    case.tear_down()


@fixture
def small_input_problem(
    problem: DerivativesTestProblem, max_input_numel: int = 100
) -> DerivativesTestProblem:
    """Skip cases with large inputs.

    Args:
        problem: Test case with deterministically constructed attributes.
        max_input_numel: Maximum input size. Default: ``100``.

    Yields:
        Instantiated test case with small input.
    """
    if problem.input.numel() > max_input_numel:
        skip("Input is too large:" + f" {problem.input.numel()} > {max_input_numel}")
    else:
        yield problem


@fixture
def no_loss_problem(
    small_input_problem: DerivativesTestProblem,
) -> DerivativesTestProblem:
    """Skip cases that are loss functions.

    Args:
        small_input_problem: Test case with small input.

    Yields:
        Instantiated test case that is not a loss layer.
    """
    if small_input_problem.is_loss():
        skip("Only required for non-loss layers.")
    else:
        yield small_input_problem


def test_hessian_is_zero(no_loss_problem: DerivativesTestProblem) -> None:
    """Check if the input-output Hessian is (non-)zero.

    Note:
        `hessian_is_zero` is a global statement that assumes arbitrary inputs.
        It can thus happen that the Hessian diagonal is zero for the current
        input, but not in general.

    Args:
        no_loss_problem: Test case whose module is not a loss.
    """
    backpack_res = BackpackDerivatives(no_loss_problem).hessian_is_zero()
    autograd_res = AutogradDerivatives(no_loss_problem).hessian_is_zero()

    if autograd_res and not backpack_res:
        warn(
            "Autograd Hessian diagonal is zero for this input "
            " while BackPACK implementation implies inputs with non-zero Hessian."
        )
    else:
        assert backpack_res == autograd_res


@mark.parametrize("problem", LOSS_PROBLEMS, ids=LOSS_IDS)
def test_make_hessian_mat_prod(problem: DerivativesTestProblem) -> None:
    """Test hessian_mat_prod.

    Args:
        problem: test problem
    """
    problem.set_up()
    mat = rand(4, *problem.input_shape, device=problem.device)

    autograd_res = AutogradDerivatives(problem).hessian_mat_prod(mat)
    backpack_res = BackpackDerivatives(problem).hessian_mat_prod(mat)

    check_sizes_and_values(backpack_res, autograd_res)
