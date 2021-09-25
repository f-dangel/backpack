"""Utility functions to test `backpack.core.derivatives`."""
from test.core.derivatives import derivatives_for
from typing import Tuple, Type

import torch
from torch import Tensor
from torch.nn import Module

from backpack.core.derivatives.basederivatives import BaseDerivatives


def get_available_devices():
    """Return CPU and, if present, GPU device.

    Returns:
        [torch.device]: Available devices for `torch`.
    """
    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    return devices


def derivative_cls_for(module_cls: Type[Module]) -> Type[BaseDerivatives]:
    """Return the associated derivative class for a module.

    Args:
        module_cls: Layer class.

    Returns:
        Class implementing the derivatives for `module_cls`.

    Raises:
        KeyError: if derivative for module is missing
    """
    try:
        return derivatives_for[module_cls]
    except KeyError as e:
        raise KeyError(
            f"No derivative available for {module_cls}. "
            + f"Known mappings:\n{derivatives_for}"
        ) from e


def classification_targets(size: Tuple[int, ...], num_classes: int) -> Tensor:
    """Create random targets for classes 0, ..., `num_classes - 1`.

    Args:
        size: shape of targets
        num_classes: number of classes

    Returns:
        classification targets
    """
    return torch.randint(size=size, low=0, high=num_classes)


def regression_targets(size: Tuple[int, ...]) -> Tensor:
    """Create random targets for regression.

    Args:
        size: shape of targets

    Returns:
        regression targets
    """
    return torch.rand(size=size)
