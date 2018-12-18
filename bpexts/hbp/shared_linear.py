"""Hessian backpropagation for multiple linear layers sharing the input.

This allows to treat weights/bias independently during optimization,
leading to smaller Hessians, thereby allowing for in principle massively
parallel optimization."""


from numpy import cumsum
from torch import (cat, zeros)
from torch.nn import Module
from .module import hbp_decorate
from .linear import HBPLinear


class HBPSharedLinear(hbp_decorate(Module)):
    """Linear layers sharing the same input.

    The output of all layers is concatenated to yield the module's
    output.
    """
    def __init__(self, in_features, out_features_list, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features_list = out_features_list
        self.out_features = sum(out_features_list)
        self.has_bias = bias
        self.create_layers()

    def create_layers(self):
        """Add linear layers all processing the same input.

        Naming convention:
        ------------------
        Submodules are registered via 'hbp_linear0', 'hbp_linear1', ...
        """
        for idx, out in enumerate(self.out_features_list):
            name = 'hbp_linear{}'.format(idx)
            layer = HBPLinear(in_features=self.in_features,
                              out_features=out,
                              bias=self.has_bias)
            self.add_module(name, layer)

    def get_submodule(self, idx):
        """Return the appropriate submodule."""
        return self.__getattr__('hbp_linear{}'.format(idx))

    # override
    def forward(self, input):
        """Apply each linear layer separately, concatenate results."""
        split_output = [layer(input) for layer in self.children()]
        return self.concatenate_output(split_output)

    def split_output_idx(self):
        """Return the indices where quantities are split in output."""
        idx = [0] + list(cumsum(self.out_features_list))
        return idx

    def concatenate_output(self, split_output):
        """Concatenate output of each linear layer into a large vector."""
        return cat(split_output, dim=1)

    # override
    def hbp_hooks(self):
        """No additional hooks are required.

        Note: Potentially, the input could be shared among the layers,
        reducing memory consumption (each layer saves the input when
        HBP is enabled."""
        pass

    # override
    def backward_hessian(self, output_hessian, compute_input_hessian=True,
                         modify_2nd_order_terms='none'):
        """Compute Hessian w.r.t. input from Hessian w.r.t. output.

        The output Hessian has to be split into diagonal blocks, where
        the size of a block corresponds to the number of outputs from
        the associated layer.

        After backwarding the blocks through each block separately,
        they summed.

        """
        # split into blocks
        split_out_hessians = self.split_output_hessian(output_hessian)
        # input hessian
        if compute_input_hessian is True:
            in_hessian = zeros(self.in_features, self.in_features)
        # call backward_hessian for each layer separately
        for layer, split_out in zip(self.children(), split_out_hessians):
            in_h = layer.backward_hessian(
                    split_out,
                    compute_input_hessian=compute_input_hessian,
                    modify_2nd_order_terms=modify_2nd_order_terms)
            if compute_input_hessian is True:
                in_hessian.add_(in_h)
        if compute_input_hessian is True:
            return in_hessian

    def split_output_hessian(self, output_hessian):
        """Cut out diagonal blocks corresponding to layer output sizes."""
        idx = self.split_output_idx()
        return [output_hessian[idx[i]:idx[i + 1], idx[i]:idx[i + 1]]
                for i in range(len(self.out_features_list))]
