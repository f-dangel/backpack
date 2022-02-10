"""Contains errors for BackPACK."""
from typing import Union
from warnings import warn

from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d


def batch_norm_raise_error_if_train(
    module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d], raise_error: bool = True
) -> None:
    """Check if BatchNorm module is in training mode.

    Args:
        module: BatchNorm module to check
        raise_error: whether to raise an error, alternatively warn. Default: True.

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
        )
        if raise_error:
            raise NotImplementedError(message)
        else:
            warn(message)
