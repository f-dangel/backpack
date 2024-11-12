"""Kronecker approximations of the Hessian for 2d convolution layers."""

from backpack.extensions.secondorder.hbp.convnd import HBPConvNd


class HBPConv2d(HBPConvNd):
    """Compute Kronecker-structured Hessian approximations for 2d convolutions."""

    def __init__(self):
        """Instantiate base class with convolution dimension."""
        super().__init__(N=2)
