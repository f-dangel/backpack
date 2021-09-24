"""Contains utility functions."""
from typing import Type

from pkg_resources import get_distribution, packaging

TORCH_VERSION = packaging.version.parse(get_distribution("torch").version)
TORCH_VERSION_AT_LEAST_1_7_0 = TORCH_VERSION >= packaging.version.parse("1.7.0")
TORCH_VERSION_AT_LEAST_1_8_0 = TORCH_VERSION >= packaging.version.parse("1.8.0")
TORCH_VERSION_AT_LEAST_1_9_0 = TORCH_VERSION >= packaging.version.parse("1.9.0")
TORCH_VERSION_AT_LEAST_1_9_1 = TORCH_VERSION >= packaging.version.parse("1.9.1")
TORCH_VERSION_AT_LEAST_2_0_0 = TORCH_VERSION >= packaging.version.parse("2.0.0")

FULL_BACKWARD_HOOK: bool = TORCH_VERSION_AT_LEAST_1_9_0
CONVERTER_AVAILABLE: bool = TORCH_VERSION_AT_LEAST_1_9_0
ADAPTIVE_AVG_POOL_BUG: bool = not TORCH_VERSION_AT_LEAST_2_0_0


def exception_inside_backward_pass(error: Type[Exception]) -> Type[Exception]:
    """Returns the type of exception that gets raised inside a backward pass by PyTorch.

    For Torch>=1.7.0 the error is identical.

    Args:
        error: previous exception type

    Returns:
        new exception type
    """
    if TORCH_VERSION_AT_LEAST_1_7_0:
        return error
    else:
        return RuntimeError
