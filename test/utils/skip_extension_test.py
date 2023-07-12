"""Contains skip conditions for BackPACK's extension tests."""

from test.extensions.problem import ExtensionsTestProblem

from pytest import skip
from torch.nn import BCEWithLogitsLoss


def skip_BCEWithLogitsLoss_non_binary_labels(problem: ExtensionsTestProblem) -> None:
    """Skip if case uses BCEWithLogitsLoss as loss function with non-binary labels.

    Args:
        problem: Extension test case.
    """
    if isinstance(problem.loss_function, BCEWithLogitsLoss) and any(
        y not in [0, 1] for y in problem.target.flatten()
    ):
        skip("Skipping BCEWithLogitsLoss with non-binary labels")


def skip_BCEWithLogitsLoss(problem: ExtensionsTestProblem) -> None:
    """Skip if case uses BCEWithLogitsLoss as loss function.

    Args:
        problem: Extension test case.
    """
    if isinstance(problem.loss_function, BCEWithLogitsLoss):
        skip("Skipping BCEWithLogitsLoss")
