"""Partial derivatives for `torch.nn.ConvTranspose2d`."""

from backpack.core.derivatives.conv_transposend import ConvTransposeNDDerivatives


class ConvTranspose2DDerivatives(ConvTransposeNDDerivatives):
    def __init__(self):
        super().__init__(N=2)
