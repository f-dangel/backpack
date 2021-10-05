"""Implements the derivatives for AdaptiveAvgPool."""
from typing import List, Tuple, Union
from warnings import warn

from torch import Size
from torch.nn import AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d

from backpack.core.derivatives.avgpoolnd import AvgPoolNDDerivatives
from backpack.utils import ADAPTIVE_AVG_POOL_BUG


class AdaptiveAvgPoolNDDerivatives(AvgPoolNDDerivatives):
    """Implements the derivatives for AdaptiveAvgPool."""

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
        if ADAPTIVE_AVG_POOL_BUG and module.input0.is_cuda and (self.N == 3):
            warn(
                "Be careful when computing gradients of AdaptiveAvgPool3d. "
                "There is a bug using autograd.grad on cuda with AdaptiveAvgPool3d. "
                "https://discuss.pytorch.org/t/bug-report-autograd-grad-adaptiveavgpool3d-cuda/124614 "  # noqa: B950
                "BackPACK derivatives are correct."
            )

        shape_input: Size = module.input0.shape
        shape_output: Size = module.output.shape

        # check length of input shape
        if not len(shape_input) == (self.N + 2):
            raise NotImplementedError(
                f"input must be (batch_size, C, ...) with ... {self.N} dimensions"
            )

        # check if input shape is multiple of output shape
        if any(shape_input[2 + n] % shape_output[2 + n] != 0 for n in range(self.N)):
            raise NotImplementedError(
                f"No equivalent AvgPool (unadaptive): Input shape ({shape_input}) "
                f"must be multiple of output shape ({shape_output})."
            )

    def get_avg_pool_parameters(
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
        padding: List[int] = [0 for _ in range(self.N)]

        return stride, kernel_size, padding


class AdaptiveAvgPool1dDerivatives(AdaptiveAvgPoolNDDerivatives):
    """Derivatives for AdaptiveAvgPool1d."""

    def __init__(self):
        """Initialization."""
        super().__init__(N=1)


class AdaptiveAvgPool2dDerivatives(AdaptiveAvgPoolNDDerivatives):
    """Derivatives for AdaptiveAvgPool2d."""

    def __init__(self):
        """Initialization."""
        super().__init__(N=2)


class AdaptiveAvgPool3dDerivatives(AdaptiveAvgPoolNDDerivatives):
    """Derivatives for AdaptiveAvgPool3d."""

    def __init__(self):
        """Initialization."""
        super().__init__(N=3)
