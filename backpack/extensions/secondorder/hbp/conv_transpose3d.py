"""Kronecker approximations of the Hessian for 3d transpose convolution layers."""

from backpack.extensions.secondorder.hbp.conv_transposend import HBPConvTransposeNd


class HBPConvTranspose3d(HBPConvTransposeNd):
    """Compute Kronecker-structured Hessian proxies for 3d transpose convolutions."""

    def __init__(self):
        """Instantiate base class with convolution dimension."""
        super().__init__(N=3)
