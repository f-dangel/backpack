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
