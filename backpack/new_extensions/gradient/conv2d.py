from backpack.core.derivatives.conv2d import (Conv2DDerivatives,
                                              Conv2DConcatDerivatives)

from backpack.new_extensions.gradient.base import GradBaseModule


class GradConv2d(GradBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivatives(),
            params=["bias", "weight"]
        )


class GradConv2dConcat(GradBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DConcatDerivatives(),
            params=["weight"]
        )
