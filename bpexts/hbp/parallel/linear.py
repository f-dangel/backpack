"""Parallel series of linear layers."""

from torch import cat
from ..linear import HBPLinear
from .parallel import HBPParallel


class HBPParallelLinear(HBPParallel):
    """Handle backpropagation for a parallel series of linear layers.

    The buffer `mean_input` is maintained in the first of all
    parallel modules, with a reference to it being stored in
    the other children.
    """
    contained_class = HBPLinear

    def __init__(self, layer, num_blocks):
        if not layer.__class__ == self.contained_class:
            raise ValueError('Expecting layer of type {}, got {}'
                             .format(self.contained_class,
                                     layer.__class__))
        super().__init__(min(num_blocks, layer.out_features))

        # disable exts hooks, buffers
        layer.disable_exts()

        # convert weight from parameter to buffer
        temp_weight = layer.weight.data
        del layer.weight
        layer.register_buffer('weight', temp_weight)

        # convert bias from parameter to buffer
        temp_bias = None if layer.bias is None else layer.bias.data
        del layer.bias
        layer.register_buffer('bias', temp_bias)

        # register wrapped module
        self._register_wrapped_module(layer)

        chunked_weight = layer.weight.chunk(self.num_blocks, 0)
        chunked_bias = self.num_blocks * [None]\
                if layer.bias is None\
                else layer.bias.chunk(self.num_blocks, 0)

        for idx, (chunk_w, chunk_b) in enumerate(zip(chunked_weight,
                                                     chunked_bias)):
            out_features = chunk_w.size()[0]
            parallel_idx = self.contained_class(
                    in_features=self.main.in_features,
                    out_features=out_features,
                    bias=layer.bias is not None)
            # disable hooks, buffers (except 0th layer)
            if idx != 0:
                parallel_idx.disable_exts()

            # copy weight
            parallel_idx.weight.data = chunk_w
            # copy bias
            if layer.bias is not None:
                parallel_idx.bias.data = chunk_b
            self.main.add_module(self._parallel_module_name(idx),
                                 parallel_idx)

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
        and the main layer.
        """
        mean_input = module._get_parallel_module(0).mean_input
        module.main.register_exts_buffer('mean_input', mean_input)
        for idx, mod in enumerate(module.parallel_children()):
            if idx != 0:
                mod.register_exts_buffer('mean_input', mean_input)
   # --- end of hooks ---

    # override
    def parameter_hessian(self, output_hessian):
        """Split output Hessian into blocks, compute parameter Hessian of 
        parallel modules."""
        # split output_hessian 
        out_h_split = [(output_hessian.chunk(
                          self.num_blocks, 0)[i]).chunk(
                            self.num_blocks, 1)[i]
                    for i in range(self.num_blocks)]
        # call parameter Hessian recursively
        for mod, out_h in zip(self.parallel_children(), out_h_split):
            mod.parameter_hessian(out_h)

   # override
    def forward(self, input):
        """Feed through each parallel layer, concatenate result."""
        return cat([layer(input) for layer in self.parallel_children()], 1)
