from backpack.core.derivatives.avgpoolnd import AvgPoolNDDerivatives


class AvgPool2DDerivatives(AvgPoolNDDerivatives):
    def __init__(self):
        super().__init__(N=2)
