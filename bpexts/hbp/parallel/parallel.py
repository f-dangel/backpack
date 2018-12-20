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
    """Multiple layers acting in parallel on the same input.

    The current implementation assumes 1d-data (2d-input/output).
    """

    # naming convention for parallel modules
    layer_names = 'hbp_parallel'

    def __init__(self, layers):
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
    def hbp_hooks(self):
        """No additional hooks are required."""
        pass

    # override
    def backward_hessian(self, output_hessian, compute_input_hessian=True,
                         modify_2nd_order_terms='none'):
        """Compute Hessian w.r.t. input from Hessian w.r.t. output.

        The output Hessian has to be split into diagonal blocks, where
        the size of a block corresponds to the number of outputs from
        the associated layer.

        After backwarding the blocks through each block separately,
        they are summed.

        """
        # split into blocks
        split_out_hessians = self.split_output_hessian(output_hessian)
        # input hessian
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

    def split_output_hessian(self, output_hessian):
        """Cut diagonal blocks corresponding to layer output sizes."""
        idx = self.split_output_idx()
        return [output_hessian[idx[i]:idx[i + 1], idx[i]:idx[i + 1]]
                for i in range(len(self.out_features_list))]
