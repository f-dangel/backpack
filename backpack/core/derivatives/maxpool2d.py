from backpack.core.derivatives.maxpoolnd import MaxPoolNDDerivatives


class MaxPool2DDerivatives(MaxPoolNDDerivatives):
    def __init__(self):
        super().__init__(N=2)
