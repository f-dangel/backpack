"""Partial derivatives for `torch.nn.ConvTranspose3d`."""

from backpack.core.derivatives.conv_transposend import ConvTransposeNDDerivatives


class ConvTranspose3DDerivatives(ConvTransposeNDDerivatives):
    def __init__(self):
        super().__init__(N=3)
