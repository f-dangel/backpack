"""
Hessian backpropagation implementation of multiple modules connected
in parallel to the same input.

Convert layers to parallel series and unite their parameters.

Both versions behave identically in the forward and backward
pass, while the backward pass of the Hessian yields the Hessian
for different groupings of parameters.
"""

from torch.nn import Module
from numpy import cumsum
from torch import (cat, zeros)
from ..module import hbp_decorate
from ..linear import HBPLinear
from ..combined import HBPCompositionActivationLinear
from .converter_linear import HBPParallelConverterLinear
from .converter_combined import HBPParallelConverterCompositionActivationLinear


class HBPParallel(hbp_decorate(Module)):
    """Multiple layers in parallel, all of the same type.

    Perform conversion from a usual HBP module to a parallel one,
    take care of parameter splitting and uniting.
    """
    # naming convention for parallel modules
    layer_names = 'hbp_parallel'

    def __init__(self, layers):
        super().__init__()

        different_classes = set([l.__class__ for l in layers])
        if len(different_classes) != 1:
            raise ValueError('Expecting layers of same type,'
                             ' got {}'.format(different_classes))
        self.layer_class = different_classes.pop()

        self.out_features_list = None
        for idx, layer in enumerate(layers):
            self.register_parallel_layer(layer, idx)

    def register_parallel_layer(self, layer, idx):
        """Register a layer acting in parallel.

        Modules are named `hbp_parallel{idx}`.

        Parameters:
        -----------
        layer : (HBPModule)
            Module
        idx : (int)
            Index for identification
        """
        name = '{}{}'.format(self.layer_names, idx)
        self.add_module(name, layer)

    def get_submodule(self, idx):
        """Return the appropriate submodule."""
        return self.__getattr__('{}{}'.format(self.layer_names, idx))

    @classmethod
    def from_module(cls, layer):
        """Convert HBPModule to HBPParallel with a single submodule.

        Parameters:
        -----------
        layer : (HBPModule)
            Module to be converted to a parallel series

        Returns:
        --------
        (HBPParallel)
            Parallel series consisting with only one submodule
        """
        return cls([layer])

    # override
    def forward(self, input):
        """Apply each module separately, concatenate results.

        Shapes of the output are stored in `self.out_features_list`.
        Shape of the input is stored in `self.in_features`.
        """
        split_output = [layer(input) for layer in self.children()]
        self.in_features = input.size()[1]
        self.out_features_list = [out.size()[1] for out in split_output]
        return self.concatenate_output(split_output)

    def split_output_idx(self):
        """Return the indices where quantities are split in output."""
        idx = [0] + list(cumsum(self.out_features_list))
        return idx

    def concatenate_output(self, split_output):
        """Concatenate output of each layer into a large vector."""
        return cat(split_output, dim=1)

    # override
    def hbp_hooks(self):
        """No additional hooks are required."""
        pass

    # override
    def backward_hessian(self, output_hessian,
                         compute_input_hessian=True,
                         compute_param_hessian=True,
                         modify_2nd_order_terms='none',
                         input_hessian_mode='exact'):
        """Compute Hessian w.r.t. input from Hessian w.r.t. output.

        The output Hessian has to be split into diagonal blocks, where
        the size of a block corresponds to the number of outputs from
        the associated layer.

        Parameters:
        -----------
        input_hessian_mode : (str)
            `"exact"` or `"blockwise"`, strategy for computing the
            Hessian with respect to the layer's input:
            * Exact: The input Hessian is computed as if no layer
                     splitting was present
            * Blockwise: The output Hessian is split into diagonal
                         blocks corresponding to the outputs of the
                         parallelized layers, each block is then
                         backwarded throught its creating layer and
                         the resulting Hessians are summed over the
                         parallel layers.
        """
        # split into blocks
        split_out_hessians = self.split_output_hessian(output_hessian)

        # input Hessian with blockwise strategy
        if input_hessian_mode == 'blockwise':
            if compute_input_hessian is True:
                in_hessian = None
            # call backward_hessian for each layer separately
            for layer, split_out in zip(self.children(), split_out_hessians):
                in_h = layer.backward_hessian(
                        split_out,
                        compute_input_hessian=compute_input_hessian,
                        modify_2nd_order_terms=modify_2nd_order_terms)
                if compute_input_hessian is True:
                    if in_hessian is None:
                        in_hessian = in_h
                    else:
                        in_hessian.add_(in_h)
            if compute_input_hessian is True:
                return in_hessian

        # input Hessian with exact strategy
        elif input_hessian_mode == 'exact':
            self.backward_hessian(
                    output_hessian,
                    compute_input_hessian=False,
                    compute_param_hessian=compute_param_hessian,
                    modify_2nd_order_terms=modify_2nd_order_terms,
                    input_hessian_mode='blockwise')
            # call backward_hessian for each layer separately
            united = self.unite()
            return united.backward_hessian(
                    output_hessian,
                    compute_input_hessian=compute_input_hessian,
                    # only compute param Hessian in blockwise mode
                    compute_param_hessian=False,
                    modify_2nd_order_terms=modify_2nd_order_terms,
                    input_hessian_mode='blockwise')
        else:
            raise ValueError('Unknown input_hessian_mode: {}'
                             .format(input_hessian_mode))

    def split_output_hessian(self, output_hessian):
        """Cut diagonal blocks corresponding to layer output sizes."""
        idx = self.split_output_idx()
        return [output_hessian[idx[i]:idx[i + 1], idx[i]:idx[i + 1]]
                for i in range(len(self.out_features_list))]

    def unite(self):
        """Unite all parallel layers into a single one."""
        converter = self.get_converter()
        layer = converter.unite(self)
        parallel = self.__class__([layer])
        parallel.out_features_list = None\
            if self.out_features_list is None\
            else [sum(self.out_features_list)]
        return parallel

    def split(self, out_features_list):
        """Split layer into multiple parallel ones."""
        united = self.unite()
        converter = self.get_converter()
        layers = converter.split(united, out_features_list)
        parallel = self.__class__(layers)
        parallel.out_features_list = out_features_list
        return parallel

    def split_into_blocks(self, num_blocks):
        """Split layer into `num_blocks` parallel modules."""
        out_features_list = self.compute_out_features_list(num_blocks)
        return self.split(out_features_list)

    def get_converter(self):
        """Return the appropriate converter for layers."""
        if self.layer_class is HBPLinear:
            return HBPParallelConverterLinear
        elif issubclass(self.layer_class, HBPCompositionActivationLinear):
            return HBPParallelConverterCompositionActivationLinear
        else:
            raise ValueError('No conversion known for layer of type '
                             '{}'.format(self.layer_class))

    def total_out_features(self):
        """Return the number of out_features in total."""
        if self.layer_class is HBPLinear:
            return sum(mod.out_features for mod in self.children())
        elif issubclass(self.layer_class, HBPCompositionActivationLinear):
            return sum(mod.linear.out_features for mod in self.children())
        else:
            raise ValueError('No method for getting outputs known for layer '
                             '{}'.format(self.layer_class))

    def compute_out_features_list(self, num_blocks):
        """Compute the sizes of the output when splitting into blocks."""
        out_features = self.total_out_features()
        if num_blocks <= 0:
            raise ValueError('Parameter splitting only valid for'
                             ' non-negative number of blocks, but '
                             ' got {}'.format(num_blocks))
        num_blocks = min(out_features, num_blocks)
        block_size, block_rest = divmod(out_features, num_blocks)
        out_features_list = num_blocks * [block_size]
        if block_rest != 0:
            for i in range(block_rest):
                out_features_list[i] += 1
        return list(out_features_list)
