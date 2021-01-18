from backpack.core.derivatives.maxpoolnd import MaxPoolNDDerivatives


class MaxPool1DDerivatives(MaxPoolNDDerivatives):
    def __init__(self):
        super().__init__(N=1)
