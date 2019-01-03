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


class HBPParallel(hbp_decorate(Module)):
    """Multiple layers in parallel.

    Perform conversion from a usual HBP module to a parallel one,
    take care of parameter splitting and uniting.

    The user has to specify the `unite` method in order to get a
    working backward pass for the Hessian.
    """
    # naming convention for parallel modules
    layer_names = 'hbp_parallel'

    def __init__(self, *layers):
        super().__init__()
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
    def backward_hessian(self, output_hessian,
                         compute_input_hessian=True,
                         modify_2nd_order_terms='none'):
        """Compute Hessian w.r.t. input from Hessian w.r.t. output.

        The output Hessian has to be split into diagonal blocks, where
        the size of a block corresponds to the number of outputs from
        the associated layer.
        """
        # split into blocks
        split_out_hessians = self.split_output_hessian(output_hessian)
        # compute parameter Hessians blockwise
        for layer, split_out in zip(self.children(), split_out_hessians):
            layer.parameter_hessian(split_out)
        # input Hessian
        if compute_input_hessian is False:
            return None
        else:
            united = self.unite()
            return united.get_submodule(0).input_hessian(
                    output_hessian,
                    modify_2nd_order_terms=modify_2nd_order_terms)

    def split_output_hessian(self, output_hessian):
        """Cut diagonal blocks corresponding to layer output sizes."""
        idx = self.split_output_idx()
        return [output_hessian[idx[i]:idx[i + 1], idx[i]:idx[i + 1]]
                for i in range(len(self.out_features_list))]

    def unite(self):
        """Unite all parallel layers into a single one."""
        raise NotImplementedError

    def split(self, out_features_list):
        """Split layer into multiple parallel ones."""
        raise NotImplementedError

    def split_into_blocks(self, num_blocks):
        """Split layer into `num_blocks` parallel modules."""
        out_features_list = self.compute_out_features_list(num_blocks)
        return self.split(out_features_list)

    def compute_out_features_list(self, num_blocks):
        """Compute the sizes of the output when splitting into blocks."""
        out_features = sum(self.out_features_list)
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
