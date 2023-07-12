"""Kronecker approximations of the Hessian for 3d convolution layers."""

from backpack.extensions.secondorder.hbp.convnd import HBPConvNd


class HBPConv3d(HBPConvNd):
    """Computes Kronecker-structured Hessian approximations for 3d convolutions."""

    def __init__(self):
        """Instantiate base class with convolution dimension."""
        super().__init__(N=3)
