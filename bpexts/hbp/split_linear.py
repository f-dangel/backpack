"""Hessian backpropagation for a linear layer with split input."""

from numpy import cumsum
from torch import (cat, zeros)
from torch.nn import Module
from .module import hbp_decorate
from .linear import HBPLinear


class HBPSplitLinear(hbp_decorate(Module)):
    """Linear layers processing split input.

    The input to the layer is split into smaller junks,
    each of them being processed by a linear layer.

    Parameters:
    -----------
    in_features_list (list(int)): List containing the dimension of the
                            split. Sums up to total dimension of input
    out_features_list (list(int)): List containing the dimension of the
                            split. Sums up to total dimension of output.
    bias (bool): Use a bias term for the linear layers
    """
    def __init__(self, in_features_list, out_features_list, bias=True):
        super().__init__()
        self.in_features_list = in_features_list
        self.in_features = sum(in_features_list)
        self.out_features_list = out_features_list
        self.out_features = sum(out_features_list)
        self.has_bias = bias
        self.create_layers()

    def create_layers(self):
        """Add linear layers processing the split input.

        Naming convention:
        ------------------
        Submodules are registered via 'hbp_linear0', 'hbp_linear1', ...
        """
        for idx, (in_, out) in enumerate(zip(self.in_features_list,
                                             self.out_features_list)):
            name = 'hbp_linear{}'.format(idx)
            layer = HBPLinear(in_features=in_,
                              out_features=out,
                              bias=self.has_bias)
            self.add_module(name, layer)

    def get_submodule(self, idx):
        """Return the appropriate submodule."""
        return self.__getattr__('hbp_linear{}'.format(idx))

    # override
    def forward(self, input):
        """Split input, apply linear layer to each, concatenate results."""
        split_input = self.split_input(input)
        split_output = [layer(split_in)
                        for split_in, layer in zip(split_input,
                                                   self.children())]
        return self.concatenate_output(split_output)

    def split_input(self, input):
        """Split input into chunks that will be processed separately."""
        idx = self.split_input_idx()
        return [input[:, idx[i]:idx[i + 1]]
                for i in range(len(self.in_features_list))]

    def split_input_idx(self):
        """Return the indices where quantities are split in input."""
        idx = [0] + list(cumsum(self.in_features_list))
        return idx

    def split_output_idx(self):
        """Return the indices where quantities are split in output."""
        idx = [0] + list(cumsum(self.out_features_list))
        return idx

    def concatenate_output(self, split_output):
        """Concatenate output of each linear layer into a large vector."""
        return cat(split_output, dim=1)

    # override
    def hbp_hooks(self):
        """No additional hooks are required."""
        pass

    # override
    def backward_hessian(self, output_hessian,
                         compute_input_hessian=True,
                         compute_param_hessian=True,
                         modify_2nd_order_terms='none'):
        """Compute Hessian w.r.t. input from Hessian w.r.t. output.

        The output Hessian has to be split into diagonal blocks, where
        the size of a block corresponds to the number of outputs from
        the associated layer.

        After backwarding the blocks through each block separately,
        they are concatenated into a large block-diagonal matrix again.

        Note:
        -----
        Future plan: Use sparse matrix (BDA) as soon as PyTorch API
        state changes from experimental to fix.
        """
        # split into blocks, collect backwarded blocks in list
        split_out_hessians = self.split_output_hessian(output_hessian)
        split_in_hessians = []
        # call backward_hessian for each layer separately
        for layer, split_out in zip(self.children(), split_out_hessians):
            split_in_hessians.append(
                    layer.backward_hessian(
                        split_out,
                        compute_input_hessian=compute_input_hessian,
                        compute_param_hessian=compute_param_hessian,
                        modify_2nd_order_terms=modify_2nd_order_terms))
        # concatenate into large block-diagonal matrix
        if compute_input_hessian is True:
            in_hessian = self.concatenate_input_hessians(split_in_hessians)
            return in_hessian

    def split_output_hessian(self, output_hessian):
        """Cut out diagonal blocks corresponding to layer output sizes."""
        idx = self.split_output_idx()
        return [output_hessian[idx[i]:idx[i + 1], idx[i]:idx[i + 1]]
                for i in range(len(self.in_features_list))]

    def concatenate_input_hessians(self, split_in_hessians):
        """BDA matrix with blocks corresponding to split input Hessians."""
        idx = self.split_input_idx()
        input_hessian = zeros(self.in_features, self.in_features)
        # fill in blocks
        for i, block in enumerate(split_in_hessians):
            input_hessian[idx[i]:idx[i + 1], idx[i]:idx[i + 1]] = block
        return input_hessian
