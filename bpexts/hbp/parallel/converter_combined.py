"""Converter from HBPCompositionActivationLinear to parallel series."""

from numpy import cumsum
from torch import cat
from warnings import warn
from .converter import HBPParallelConverter
from ..combined import HBPCompositionActivationLinear


class HBPParallelConverterCompositionActivationLinear(HBPParallelConverter):
    """Convert single/multiple parallel series of HBPComposition."""
    @staticmethod
    def split(parallel, out_features_list):
        """HBPComposition layer to multiple parallel linear layers.

        Parameters:
        -----------
        parallel : (HBPParallelIdentical)
            Parallel series with a single HBPComposition layer
        out_features_list : (list(int))
            Output features for each of the parallel modules

        Returns:
        --------
        (list(HBPLinear))
        """
        composition = parallel.get_submodule(0)
        if not sum(out_features_list) == composition.linear.out_features:
            raise ValueError('Invalid splitting: {} does not sum'
                             'to {}'.format(out_features_list,
                                            composition.linear.out_features))

        layers = []
        in_features = composition.linear.in_features
        has_bias = (composition.linear.bias is not None)
        idx = [0] + list(cumsum(out_features_list))
        idx = [(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]
        for out, (i, j) in zip(out_features_list, idx):
            layer = composition.__class__(in_features=in_features,
                                          out_features=out,
                                          bias=has_bias)
            layer.linear.weight.data = composition.linear.weight.data[i:j, :]
            if has_bias:
                layer.linear.bias.data = composition.linear.bias.data[i:j]
            layers.append(layer)

        # TODO: Buffers

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
        out_features = sum(mod.linear.out_features
                           for mod in parallel.children())

        in_features = set(mod.linear.in_features
                          for mod in parallel.children())
        if not len(in_features) == 1:
            raise ValueError('Expect same in_features, got {}'
                             .format(in_features))
        in_features = in_features.pop()

        has_bias = set(mod.linear.bias is not None
                       for mod in parallel.children())
        if not len(has_bias) == 1:
            raise ValueError('Expect simultaneous presence/absence'
                             ' of bias, got {}'.format(has_bias))
        has_bias = has_bias.pop()

        layer = parallel.layer_class(in_features=in_features,
                                     out_features=out_features,
                                     bias=has_bias)

        # concatenate weight matrix and assign
        weight = None
        for mod in parallel.children():
            weight = mod.linear.weight.data if weight is None\
                    else cat([weight, mod.linear.weight.data])
        layer.linear.weight.data = weight

        # concatenate bias and assign
        if has_bias:
            bias = None
            for mod in parallel.children():
                bias = mod.linear.bias.data if bias is None\
                        else cat([bias, mod.linear.bias.data])
            layer.linear.bias.data = bias

        # buffer grad_output
        try:
            grad_output = None
            for mod in parallel.children():
                grad_output = mod.grad_output if grad_output is None\
                        else cat([grad_output, mod.grad_output])
            layer.register_exts_buffer('grad_output', grad_output)
        except AttributeError:
            warn('Could not copy/find buffer grad_output')

        # buffer grad_phi
        try:
            grad_phi = None
            for mod in parallel.children():
                grad_phi = mod.grad_phi if grad_phi is None\
                        else cat([grad_phi, mod.grad_phi])
            layer.register_exts_buffer('grad_phi', grad_phi)
        except AttributeError:
            warn('Could not copy/find buffer grad_phi')

        # buffer gradgrad_phi
        try:
            gradgrad_phi = None
            for mod in parallel.children():
                gradgrad_phi = mod.gradgrad_phi if gradgrad_phi is None\
                        else cat([gradgrad_phi, mod.gradgrad_phi])
            layer.register_exts_buffer('gradgrad_phi', gradgrad_phi)
        except AttributeError:
            warn('Could not copy/find buffer gradgrad_phi')

        return layer
