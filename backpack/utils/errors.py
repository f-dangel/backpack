"""Contains errors for BackPACK."""
from typing import Union

from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d


def batch_norm_raise_error_if_train(
    module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]
) -> None:
    """Check if BatchNorm module is in training mode.

    Args:
        module: BatchNorm module to check

    Raises:
        NotImplementedError: if module is in training mode
    """
    if module.training:
        raise NotImplementedError(
            "There is a BatchNorm module in training mode. "
            "It is possible to compute BackPACK quantities in training mode."
            "Be aware that quantities might be different, i.e. individual gradients"
            "are not well defined anymore."
            "If you want to compute quantities in training mode delete this error."
        )
