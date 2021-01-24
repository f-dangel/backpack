from backpack.core.derivatives.avgpoolnd import AvgPoolNDDerivatives


class AvgPool3DDerivatives(AvgPoolNDDerivatives):
    def __init__(self):
        super().__init__(N=3)
