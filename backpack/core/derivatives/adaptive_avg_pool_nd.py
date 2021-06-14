"""Implements the derivatives for AdaptiveAvgPool."""
from typing import Any, List, Tuple, Union

from torch import Size
from torch.nn import AdaptiveAvgPool1d, AdaptiveAvgPool2d

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
        self, module: Union[AdaptiveAvgPool1d, AdaptiveAvgPool2d]
    ) -> Tuple[List[int], List[int]]:
        """Checks if parameters are supported.

        Specifically checks if input shape is multiple of output shape.
        In this case, there are parameters for AvgPoolND that are equivalent.

        https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work/63603993#63603993 # noqa: B950

        Args:
            module: the module to check

        Returns:
            stride and kernel_size as lists of length self.N

        Raises:
            NotImplementedError: if target shape is not supported or not compatible
                with input shape.
        """
        shape_input: Size = module.input0.shape
        shape_target: Union[Size, tuple] = self._get_shape_target(module)

        # check length of input shape
        if not len(shape_input) == (self.N + 2):
            raise NotImplementedError(
                f"input must be (batch_size, C, ...) with ... {self.N} dimensions"
            )
        # check length of target shape
        if not len(shape_target) == self.N:
            raise NotImplementedError(
                f"shape of target should have {self.N} dimensions"
            )

        # check if shapes are multiples and calculate equivalent AvgPoolND parameters
        stride: List[int] = []
        kernel_size: List[int] = []
        for n in range(self.N):
            in_dim: int = shape_input[2 + n]
            out_dim: int = shape_target[n]
            if (in_dim % out_dim) != 0:
                raise NotImplementedError(
                    "Not supported in BackPACK: Input shape of AdaptiveAvgPool "
                    "must be multiple of output shape. Only in this case, there is "
                    "an equivalent AvgPool module."
                )
            stride.append(in_dim // out_dim)
            kernel_size.append(in_dim - (out_dim - 1) * stride[n])

        return stride, kernel_size

    def _get_shape_target(self, module) -> Union[Size, Tuple]:
        if isinstance(module.output_size, Size):
            shape_target: Size = module.output_size
        elif isinstance(module.output_size, int):
            if self.N == 1:
                shape_target: tuple = (module.output_size,)
            elif self.N == 2:
                shape_target: tuple = (module.output_size, module.output_size)
            else:
                raise NotImplementedError("Number of dimensions should be 1 or 2")
        elif isinstance(module.output_size, tuple):
            if not all([isinstance(element, int) for element in module.output_size]):
                raise NotImplementedError(
                    "Not supported in BackPACK: all elements of output_size must be int"
                )
            shape_target: tuple = module.output_size
        else:
            raise NotImplementedError(
                f"AdaptiveAveragePoolingDerivatives does not support target "
                f"output_size={module.output_size}"
            )
        return shape_target

    def check_exotic_parameters(self, module):
        """Overwrite and pass because no exotic parameters.

        Args:
            module: module
        """
        pass

    def get_parameters(self, module) -> Tuple[Any, Any, Any]:
        """Return the parameters of an equivalent AvgPoolNd.

        Args:
            module: module, not used

        Returns:
            stride, kernel_size, padding
        """
        stride, kernel_size = self.check_parameters(module)
        return stride, kernel_size, 0


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
