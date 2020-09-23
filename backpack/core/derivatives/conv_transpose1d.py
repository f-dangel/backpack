"""Partial derivatives for `torch.nn.ConvTranspose1d`."""

from backpack.core.derivatives.conv_transposend import ConvTransposeNDDerivatives


class ConvTranspose1DDerivatives(ConvTransposeNDDerivatives):
    def __init__(self):
        super().__init__(N=1)
