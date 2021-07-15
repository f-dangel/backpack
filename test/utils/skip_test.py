"""Skip specific tests."""
import pytest

from backpack.utils import TORCH_VERSION_HIGHER_THAN_1_9_1


def skip_adaptive_avg_pool3d_cuda(request) -> None:
    """Skips test if AdaptiveAvgPool3d and cuda.

    Args:
        request: problem request
    """
    if TORCH_VERSION_HIGHER_THAN_1_9_1:
        pass
    else:
        if all(
            string in request.node.callspec.id
            for string in ["AdaptiveAvgPool3d", "cuda"]
        ):
            pytest.skip(
                "Skip test because AdaptiveAvgPool3d does not work on cuda. "
                "Is fixed in torch 1.9.1."
            )
