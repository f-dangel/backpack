from backpack.core.derivatives.linear import (LinearDerivatives,
                                              LinearConcatDerivatives)
from .base import GradBaseModule


class GradLinear(GradBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=LinearDerivatives(),
            params=["bias", "weight"]
        )


class GradLinearConcat(GradBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=LinearConcatDerivatives(),
            params=["weight"]
        )
