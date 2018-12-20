"""Converter fromn HBPLinear to parallel series."""

from numpy import cumsum
from torch import cat
from .converter import HBPParallelConverter
from ..linear import HBPLinear


class HBPParallelConverterLinear(HBPParallelConverter):
    """Convert parallel series of HBPLinear layers into a single one."""
    # override
    @staticmethod
    def split(parallel, out_features_list):
        """HBPLinear layer to multiple parallel linear layers.

        Parameters:
        -----------
        parallel : (HBPParallelIdentica)
            Parallel series with a single linear layer
        out_features_list : (list(int))
            Output features for each of the parallel modules

        Returns:
        --------
        (list(HBPLinear))
        """
        linear = parallel.get_submodule(0)
        if not sum(out_features_list) == linear.out_features:
            raise ValueError('Invalid splitting: {} does not sum'
                             'to {}'.format(out_features_list,
                                            linear.out_features))

        layers = []
        in_features = linear.in_features
        has_bias = (linear.bias is not None)
        idx = [0] + list(cumsum(out_features_list))
        idx = [(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]
        for out, (i, j) in zip(out_features_list, idx):
            layer = HBPLinear(in_features=in_features,
                              out_features=out,
                              bias=has_bias)
            layer.weight.data = linear.weight.data[i:j, :]
            if has_bias:
                layer.bias.data = linear.bias.data[i:j]
            layers.append(layer)

        return layers

    # override
    @staticmethod
    def unite(parallel):
        """Convert multiple linear layers into a single one.

        Parameters:
        -----------
        parallel : (HBPParallelIdentical)
            Parallel series of linear modules

        Returns:
        --------
        (HBPLinear)
        """
        out_features = sum(mod.out_features for mod in parallel.children())

        in_features = set(mod.in_features for mod in parallel.children())
        if not len(in_features) == 1:
            raise ValueError('Expect same in_features, got {}'
                             .format(in_features))
        in_features = in_features.pop()

        has_bias = set(mod.bias is not None for mod in parallel.children())
        if not len(has_bias) == 1:
            raise ValueError('Expect simultaneous presence/absence'
                             ' of bias, got {}'.format(has_bias))
        has_bias = has_bias.pop()

        layer = HBPLinear(in_features=in_features,
                          out_features=out_features,
                          bias=has_bias)

        # concatenate weight matrix and assign
        weight = None
        for mod in parallel.children():
            weight = mod.weight.data if weight is None\
                    else cat([weight, mod.weight.data])
        layer.weight.data = weight

        # concatenate bias and assign
        if has_bias:
            bias = None
            for mod in parallel.children():
                bias = mod.bias.data if bias is None\
                        else cat([bias, mod.bias.data])
            layer.bias.data = bias

        return layer
