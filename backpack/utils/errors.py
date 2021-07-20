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
        NotImplementedError: if module is in training mode
    """
    if module.training:
        message = (
            "There is a BatchNorm module in training mode. "
            "It is possible to compute BackPACK quantities in training mode."
            "Be aware that quantities might be different, i.e. individual "
            "gradients are not well defined anymore."
            "If you want to compute quantities in training mode delete this error."
        )
        if raise_error:
            raise NotImplementedError(message)
        else:
            warn(message)
