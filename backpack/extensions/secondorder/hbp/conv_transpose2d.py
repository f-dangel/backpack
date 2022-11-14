"""Kronecker approximations of the Hessian for 2d transpose convolution layers."""

from backpack.extensions.secondorder.hbp.conv_transposend import HBPConvTransposeNd


class HBPConvTranspose2d(HBPConvTransposeNd):
    """Compute Kronecker-structured Hessian proxies for 2d transpose convolutions."""

    def __init__(self):
        """Instantiate base class with convolution dimension."""
        super().__init__(N=2)
