"""The code relies on the insight that average pooling can be understood as
convolution over single channels with a constant kernel."""

from backpack.core.derivatives.avgpoolnd import AvgPoolNDDerivatives


class AvgPool1DDerivatives(AvgPoolNDDerivatives):
    def __init__(self):
        super().__init__(N=1)
