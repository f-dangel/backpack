"""Implements the derivatives for AdaptiveAvgPool."""
from typing import List, Tuple, Union

from torch import Size
from torch.nn import AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d

from backpack.core.derivatives.avgpoolnd import AvgPoolNDDerivatives


class AdaptiveAvgPoolNDDerivatives(AvgPoolNDDerivatives):
    """Implements the derivatives for AdaptiveAvgPool."""

    def __init__(self, N: int):
        """Initialization.

        Args:
            N: number of dimensions
        """
        super(AdaptiveAvgPoolNDDerivatives, self).__init__(N=N)

    def check_parameters(
        self, module: Union[AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d]
    ) -> None:
        """Checks if the parameters are supported.

        Specifically checks if input shape is multiple of output shape.
        In this case, there are parameters for AvgPoolND that are equivalent.

        https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work/63603993#63603993 # noqa: B950

        Args:
            module: module to check

        Raises:
            NotImplementedError: if the given shapes do not match
        """
        shape_input: Size = module.input0.shape
        shape_target: Size = module.output.shape

        # check length of input shape
        if not len(shape_input) == (self.N + 2):
            raise NotImplementedError(
                f"input must be (batch_size, C, ...) with ... {self.N} dimensions"
            )
        # check length of target shape
        if not len(shape_target) == (self.N + 2):
            raise NotImplementedError(
                f"shape of target should have {self.N} dimensions"
            )

        # check if input shape is multiple of output shape
        for n in range(self.N):
            in_dim: int = shape_input[2 + n]
            out_dim: int = shape_target[2 + n]
            if (in_dim % out_dim) != 0:
                raise NotImplementedError(
                    "Not supported in BackPACK: Input shape of AdaptiveAvgPool "
                    "must be multiple of output shape. Only in this case, there is "
                    "an equivalent AvgPool module."
                )

    def get_parameters(
        self, module: Union[AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d]
    ) -> Tuple[List[int], List[int], List[int]]:
        """Return parameters for an equivalent AvgPool.

        Assumes that check_parameters has been run before.
        Therefore, does not check parameters.

        Args:
            module: module to compute on

        Returns:
            stride, kernel_size, padding as lists of length self.N
        """
        shape_input: Size = module.input0.shape
        shape_target: Size = module.output.shape

        # calculate equivalent AvgPoolND parameters
        stride: List[int] = []
        kernel_size: List[int] = []
        for n in range(self.N):
            in_dim: int = shape_input[2 + n]
            out_dim: int = shape_target[2 + n]
            stride.append(in_dim // out_dim)
            kernel_size.append(in_dim - (out_dim - 1) * stride[n])

        return stride, kernel_size, list((0,) * self.N)


class AdaptiveAvgPool1dDerivatives(AdaptiveAvgPoolNDDerivatives):
    """Derivatives for AdaptiveAvgPool1d."""

    def __init__(self):
        """Initialization."""
        super(AdaptiveAvgPool1dDerivatives, self).__init__(N=1)


class AdaptiveAvgPool2dDerivatives(AdaptiveAvgPoolNDDerivatives):
    """Derivatives for AdaptiveAvgPool2d."""

    def __init__(self):
        """Initialization."""
        super(AdaptiveAvgPool2dDerivatives, self).__init__(N=2)


class AdaptiveAvgPool3dDerivatives(AdaptiveAvgPoolNDDerivatives):
    """Derivatives for AdaptiveAvgPool3d."""

    def __init__(self):
        """Initialization."""
        super(AdaptiveAvgPool3dDerivatives, self).__init__(N=3)
