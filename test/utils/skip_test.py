"""Skip specific tests."""

from test.core.derivatives.problem import DerivativesTestProblem
from typing import List, Union

from pytest import skip
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from backpack.custom_module.permute import Permute
from backpack.utils import TORCH_VERSION_AT_LEAST_1_9_1
from backpack.utils.subsampling import get_batch_axis


def skip_adaptive_avg_pool3d_cuda(request) -> None:
    """Skips test if AdaptiveAvgPool3d and cuda.

    Args:
        request: problem request
    """
    if TORCH_VERSION_AT_LEAST_1_9_1:
        pass
    else:
        if all(
            string in request.node.callspec.id
            for string in ["AdaptiveAvgPool3d", "cuda"]
        ):
            skip(
                "Skip test because AdaptiveAvgPool3d does not work on cuda. "
                "Is fixed in torch 1.9.1."
            )


def skip_permute_with_subsampling(
    problem: DerivativesTestProblem, subsampling: Union[List[int], None]
) -> None:
    """Skip Permute module when sub-sampling is turned on.

    Permute does not assume a batch axis.

    Args:
        problem: Test case.
        subsampling: Indices of active samples.
    """
    if isinstance(problem.module, Permute) and subsampling is not None:
        skip(f"Skipping Permute with sub-sampling: {subsampling}")


def skip_batch_norm_train_mode_with_subsampling(
    problem: DerivativesTestProblem, subsampling: Union[List[int], None]
) -> None:
    """Skip BatchNorm in train mode when sub-sampling is turned on.

    Args:
        problem: Test case.
        subsampling: Indices of active samples.
    """
    if isinstance(problem.module, (BatchNorm1d, BatchNorm2d, BatchNorm3d)):
        if problem.module.train and subsampling is not None:
            skip(f"Skipping BatchNorm in train mode with sub-sampling: {subsampling}")


def skip_subsampling_conflict(
    problem: DerivativesTestProblem, subsampling: Union[List[int], None]
) -> None:
    """Skip if some samples in subsampling are not contained in input.

    Args:
        problem: Test case.
        subsampling: Indices of active samples.
    """
    N = problem.input_shape[get_batch_axis(problem.module)]
    enough_samples = subsampling is None or N > max(subsampling)
    if not enough_samples:
        skip("Not enough samples.")


def skip_no_param(problem: DerivativesTestProblem, param_str: str) -> None:
    """Skip if test case does not contain the parameter.

    Args:
        problem: Test case.
        param_str: Parameter name.
    """
    has_param = getattr(problem.module, param_str, None) is not None
    if not has_param:
        skip(f"Test case has no {param_str} parameter.")
