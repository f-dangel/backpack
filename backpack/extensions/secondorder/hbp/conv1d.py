"""Kronecker approximations of the Hessian for 1d convolution layers."""

from backpack.extensions.secondorder.hbp.convnd import HBPConvNd


class HBPConv1d(HBPConvNd):
    """Computes Kronecker-structured Hessian approximations for 1d convolutions."""

    def __init__(self):
        """Instantiate base class with convolution dimension."""
        super().__init__(N=1)
