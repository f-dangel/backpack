"""Parallel series of linear layers."""

from torch import cat
from numpy import cumsum
from warnings import warn
from ..linear import HBPLinear
from .parallel import HBPParallel


class HBPParallelLinear(HBPParallel):
    """Handle backpropagation for a parallel series of linear layers.

    The buffer `mean_input` is maintained in the first of all
    parallel modules, with a reference to it being stored in
    the other children.
    """
    contained_class = HBPLinear

    def __init__(self, *layers):
        different_classes = set(l.__class__ for l in layers)
        if not different_classes == set([HBPLinear]):
            raise ValueError('Expecting layers of type {}, got {}'
                             .format(self.contained_class,
                                     different_classes))
        for l in layers[1:]:
            l.disable_exts()
        super().__init__(*layers)

    # override
    def hbp_hooks(self):
        """Remove input hook in children, use a single copy instead."""
        self.register_exts_forward_hook(self.reference_mean_input)

    # --- hooks ---
    @staticmethod
    def reference_mean_input(module, input, output):
        """Save reference of mean_input from first child in others.

        Intended use as forward hook.
        Initialize module buffer 'mean_input' in all other children
        beside the first one.
        """
        module._reference_mean_input_in_children()

    def _reference_mean_input_in_children(self):
        """Store a reference to the buffer mean_input in each child.

        Avoid copies of the same tensor.
        """
        mean_input = self.get_submodule(0).mean_input
        for i, mod in enumerate(self.children()):
            # skip first
            if i != 0:
                mod.register_exts_buffer('mean_input',
                                         mean_input)
    # --- end of hooks ---

    def unite(self):
        """Unite all parallel children to a single one.

        Returns:
        --------
        (HBPParallelLinear)
            Parallel series of HBPLinear consisting of only a single
            child, behaves identically in forward mode.
        """
        out_features = sum(mod.out_features for mod in self.children())

        # check consistency
        in_features = set(mod.in_features for mod in self.children())
        if not len(in_features) == 1:
            raise ValueError('Expect same in_features, got {}'
                             .format(in_features))
        in_features = in_features.pop()

        # check consistency
        has_bias = set(mod.bias is not None for mod in self.children())
        if not len(has_bias) == 1:
            raise ValueError('Expect simultaneous presence/absence'
                             ' of bias, got {}'.format(has_bias))
        has_bias = has_bias.pop()

        # create HBPLinear layer
        layer = self.contained_class(in_features=in_features,
                                     out_features=out_features,
                                     bias=has_bias)

        # concatenate weight matrix and assign
        weight = cat([mod.weight.data for mod in self.children()])
        layer.weight.data = weight

        # concatenate bias and assign
        if has_bias:
            bias = cat([mod.bias.data for mod in self.children()])
            layer.bias.data = bias

        # HBPParallelLinear version with single child
        parallel = self.__class__(layer)

        # out_features_list
        parallel.out_features_list = [out_features]

        # copy over buffer of input
        try:
            mean_input = self.get_submodule(0).mean_input
            parallel.get_submodule(0).register_exts_buffer(
                    'mean_input', mean_input)
            parallel._reference_mean_input_in_children()
        except AttributeError as e:
            warn('Could not copy/find buffer mean_input.\n{}'
                 .format(e))

        return parallel

    def split(self, out_features_list):
        """Split into parallel series of HBPLinear.

        Parameters:
        -----------
        out_features_list : (list(int))
            Output features for each of the parallel modules

        Returns:
        --------
        (HBPParallelLinear)
        """
        united = self.unite()

        # check consistency
        if not sum(out_features_list) == sum(united.out_features_list):
            raise ValueError('Invalid splitting: {} does not sum'
                             'to {}'.format(out_features_list,
                                            united.out_features_list))

        # get the single HBPLinear child
        linear = united.get_submodule(0)
        in_features = linear.in_features
        has_bias = (linear.bias is not None)

        # create parallel children
        layers = []

        idx = [0] + list(cumsum(out_features_list))
        idx = [(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]
        for out, (i, j) in zip(out_features_list, idx):
            # create HBPLinear
            child = self.contained_class(in_features=in_features,
                                         out_features=out,
                                         bias=has_bias)
            # copy bias
            child.weight.data = linear.weight.data[i:j, :]
            if has_bias:
                child.bias.data = linear.bias.data[i:j]
            layers.append(child)

        # HBPParallelLinear version with single child
        parallel = self.__class__(*layers)

        # out_features_list
        parallel.out_features_list = out_features_list

        # copy over buffer of input
        try:
            mean_input = self.get_submodule(0).mean_input
            parallel.get_submodule(0).register_exts_buffer(
                    'mean_input', mean_input)
            parallel._reference_mean_input_in_children()
        except AttributeError as e:
            warn('Could not copy/find buffer mean_input.\n{}'
                 .format(e))

        return parallel
