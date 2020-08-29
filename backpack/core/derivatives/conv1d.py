from backpack.core.derivatives.convnd import ConvNDDerivatives


class Conv1DDerivatives(ConvNDDerivatives):
    def __init__(self):
        super().__init__(N=1)
