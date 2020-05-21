"""Utility functions to test `backpack.core.derivatives`"""

import torch

from backpack.core.derivatives import derivatives_for


def get_available_devices():
    """Return CPU and, if present, GPU device.

    Returns:
        [torch.device]: Available devices for `torch`.
    """
    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    return devices


def derivative_cls_for(module_cls):
    """Return the associated derivative class for a module.

    Args:
        module_cls (torch.nn.Module): Layer class.

    Returns:
        backpack.core.derivatives.Derivatives: Class implementing the
            derivatives for `module_cls`.
    """
    try:
        return derivatives_for[module_cls]
    except KeyError:
        raise KeyError(
            "No derivative available for {}".format(module_cls)
            + "Known mappings:\n{}".format(derivatives_for)
        )


def is_loss(module):
    """Return whether `module` is a `torch` loss function.

    Args:
        module (torch.nn.Module): A PyTorch module.

    Returns:
        bool: Whether `module` is a loss function.
    """
    return isinstance(module, torch.nn.modules.loss._Loss)


def classification_targets(size, num_classes):
    """Create random targets for classes 0, ..., `num_classes - 1`."""
    return torch.randint(size=size, low=0, high=num_classes)


def regression_targets(size):
    """Create random targets for regression."""
    return torch.rand(size=size)
