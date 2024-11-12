"""Skip specific tests."""

from test.core.derivatives.problem import DerivativesTestProblem
from test.extensions.problem import ExtensionsTestProblem
from typing import List, Union

from pytest import skip
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d, BCEWithLogitsLoss


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
    problem: Union[DerivativesTestProblem, ExtensionsTestProblem],
    subsampling: Union[List[int], None],
) -> None:
    """Skip if some samples in subsampling are not contained in input.

    Args:
        problem: Test case.
        subsampling: Indices of active samples.
    """
    N = problem.get_batch_size()
    enough_samples = subsampling is None or N > max(subsampling)
    if not enough_samples:
        skip("Not enough samples.")


def skip_large_parameters(
    problem: ExtensionsTestProblem, max_num_params: int = 1000
) -> None:
    """Skip architectures with too many parameters.

    Args:
        problem: Test case.
        max_num_params: Maximum number of model parameters. Default: ``1000``.
    """
    num_params = sum(p.numel() for p in problem.trainable_parameters())
    if num_params > max_num_params:
        skip(f"Model has too many parameters: {num_params} > {max_num_params}")


def skip_BCEWithLogitsLoss(problem: DerivativesTestProblem) -> None:
    """Skip if the test problem uses BCEWithLogitsLoss.

    Args:
        problem: Test case.
    """
    if isinstance(problem.module, BCEWithLogitsLoss):
        skip("Skipping BCEWithLogitsLoss")


def skip_BCEWithLogitsLoss_non_binary_labels(problem: DerivativesTestProblem) -> None:
    """Skip if the test problem uses BCEWithLogitsLoss and non-binary labels.

    Args:
        problem: Test case.
    """
    if isinstance(problem.module, BCEWithLogitsLoss) and any(
        y not in [0, 1] for y in problem.target.flatten()
    ):
        skip("Skipping BCEWithLogitsLoss with non-binary labels")
