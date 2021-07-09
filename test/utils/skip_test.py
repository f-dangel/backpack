"""Skip specific tests."""
import pytest


def skip_adaptive_avg_pool3d_cuda(request) -> None:
    """Skips test if AdaptiveAvgPool3d and cuda.

    Args:
        request: problem request
    """
    if all(
        string in request.node.callspec.id for string in ["AdaptiveAvgPool3d", "cuda"]
    ):
        pytest.skip(
            "Skip test because AdaptiveAvgPool3d does not work on cuda. "
            "Is fixed in torch 1.9.0."
        )
