"""Contains errors for BackPACK."""
from typing import Union
from warnings import warn

from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.backprop_extension import FAIL_ERROR, FAIL_WARN, FAIL_SILENT
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d


def batch_norm_raise_error_if_train(
    module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d], ext: BackpropExtension
) -> None:
    """Check if BatchNorm module is in training mode.

    Args:
        module: BatchNorm module to check
        ext: The BackpropExtension checking for errors

    Raises:
        ValueError: if module is in training mode and BackPACK extension's
            fail_mode is FAIL_ERROR (default)
    """
    if module.training:
        message = (
            "Encountered BatchNorm module in training mode."
            "Quantity to compute is undefined as BatchNorm mixes samples. "
            "You should most likely use another type of normalization. "
            "Concepts like individual gradients are not meaningful with BatchNorm. "
            "The code to compute the requested quantity may not raise an error, "
            "but the quantity will not match its definition. "
            "Advanced users: If you are specifically interested in what this code "
            "would return for a BatchNorm network, change the failure mode of "
            "the BackPACK extension (`BatchGrad`) to only raise a warning "
            "(`BatchGrad(fail_mode='WARNING')`). This is not supported behavior."
        )
        if ext._fail_mode == FAIL_ERROR:
            raise ValueError(message)
        if ext._fail_mode == FAIL_WARN:
            warn(message)
        if ext._fail_mode == FAIL_SILENT:
            return
