"""Kronecker approximations of the Hessian for 1d transpose convolution layers."""

from backpack.extensions.secondorder.hbp.conv_transposend import HBPConvTransposeNd


class HBPConvTranspose1d(HBPConvTransposeNd):
    """
    Computes Kronecker-structured Hessian approximations for 1d transpose convolutions.
    """

    def __init__(self):
        """Instantiate base class with convolution dimension."""
        super().__init__(N=1)
