from backpack.core.derivatives.maxpoolnd import MaxPoolNDDerivatives
from backpack.utils.ein import eingroup


class MaxPool3DDerivatives(MaxPoolNDDerivatives):
    def __init__(self):
        super().__init__(N=3)
