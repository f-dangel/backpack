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

    def __init__(self, layer, max_blocks):
        if not layer.__class__ == self.contained_class:
            raise ValueError('Expecting layer of type {}, got {}'
                             .format(self.contained_class,
                                     layer.__class__))
        self.max_blocks = max_blocks
        super().__init__(len(layer.weight.chunk(self.max_blocks, 0)))

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

        # register parameters as chunks of the main module buffers
        # in parallel children
        self._create_parallel_children()
        self._reference_chunks_in_parallel_children()

    def _create_parallel_children(self):
        """Can only be called after _register_wrapped_module."""
        out_features = [w.size()[1] for w in
                        self.main.weight.chunk(self.max_blocks, 0)]

        for idx, out in enumerate(out_features):
            child_name = self._parallel_module_name(idx)
            if hasattr(self.main, child_name):
                raise ValueError('Child {} already exists'
                                 .format(child_name))
            parallel_idx = self.contained_class(
                    in_features=self.main.in_features,
                    out_features=out,
                    bias=self.main.bias is not None)
            # disable hooks, buffers (except 0th layer)
            if idx != 0:
                parallel_idx.disable_exts()
            self.main.add_module(child_name, parallel_idx)

    def _reference_chunks_in_parallel_children(self):
        """ Can only be called after _create_parallel_children.
        TODO: Warn if grads are non-zero or non-None
        """
        chunked_weight = self.main.weight.chunk(self.max_blocks, 0)
        chunked_bias = self.num_blocks * [None]\
            if self.main.bias is None\
            else self.main.bias.chunk(self.max_blocks, 0)

        # checks, TODO: can be removed
        assert self.max_blocks >= self.num_blocks
        assert len(chunked_weight) == self.num_blocks
        assert len(chunked_bias) == self.num_blocks
        # ---

        for idx, (chunk_w, chunk_b, parallel) in enumerate(
                zip(chunked_weight,
                    chunked_bias,
                    self.parallel_children())):
            # copy weight
            parallel.weight.data = chunk_w
            # copy bias
            if parallel.bias is not None:
                parallel.bias.data = chunk_b

    # override
    def _apply(self, fn):
        """Need to restore references to chunked weights and bias terms after
        casting to different device/data type."""
        self = super()._apply(fn)
        # restore references
        self._reference_chunks_in_parallel_children()
        return self

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
                          self.max_blocks, 0)[i]).chunk(
                            self.max_blocks, 1)[i]
                    for i in range(self.num_blocks)]
        # call parameter Hessian recursively
        for mod, out_h in zip(self.parallel_children(), out_h_split):
            mod.parameter_hessian(out_h)

   # override
    def forward(self, input):
        """Feed through each parallel layer, concatenate result."""
        return cat([layer(input) for layer in self.parallel_children()], 1)
