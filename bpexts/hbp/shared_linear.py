"""Hessian backpropagation for multiple linear layers sharing the input.

This allows to treat weights/bias independently during optimization,
leading to smaller Hessians, thereby allowing for in principle massively
parallel optimization."""

from .linear import HBPLinear
from .parallel import HBPParallel


class HBPSharedLinear(HBPParallel):
    """Linear layers sharing the same input."""

    @classmethod
    def with_splitting(cls, in_features, out_features_list, bias=True):
        """Return linear layers acting in parallel on the same input.

        Parameters:
        -----------
        in_features : (int)
            Number of input features
        out_features_list : (list(int))
            Output features for each of the parallel modules
        bias : (bool)
            Use bias terms in linear layers

        Returns:
        --------
        (HBPParallel)
        """
        layers = []
        for idx, out in enumerate(out_features_list):
            layers.append(HBPLinear(in_features=in_features,
                                    out_features=out,
                                    bias=bias))
        return cls(layers)
