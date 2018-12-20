"""Hessian backpropagation for multiple layers sharing the input.

This allows to treat weights/bias independently during optimization,
leading to smaller Hessians, thereby allowing for in principle massively
parallel optimization.
"""


class HBPParallelConverter():
    """Convert layers to parallel series and unite their parameters.

    Both versions must behave identically in the forward and backward
    pass, while the backward pass of the Hessian yields the Hessian
    for different groupings of parameters.
    """
    @staticmethod
    def split(parallel, out_features_list):
        """Convert a layer into a parallel series of layers.

        Parameters:
        -----------
        layer : (HBPParallelIdentical)
            Parallel series with a single module
        out_features_list : (list(int))
            Output features for each of the parallel modules

        Returns:
        --------
        (list(HBPModule))
            List of modules in parallel series
        """
        raise NotImplementedError

    @staticmethod
    def unite(parallel):
        """Convert a series of parallel layers into a single one.

        Parameters:
        -----------
        layers : (list(HBPModule))
            Parallel series of modules of same type

        Returns:
        --------
        (HBPModule)
            Single layer behaving the same way as `parallel`
        """
        raise NotImplementedError
