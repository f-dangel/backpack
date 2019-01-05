"""Parallel series of HBPCompositionActivationLinear."""

from torch import cat
from numpy import cumsum
from warnings import warn
from ..combined import HBPCompositionActivationLinear
from .parallel import HBPParallel
from .linear import HBPParallelLinear


class HBPParallelCompositionActivationLinear(HBPParallel):
    """Convert single/multiple parallel series of HBPComposition."""
    contained_parent_class = HBPCompositionActivationLinear

    def __init__(self, layer, max_blocks):
        self.contained_class = layer.__class__
        if not issubclass(self.contained_class, self.contained_parent_class):
            raise ValueError('Expecting layers derived from {}, got {}'
                             .format(self.contained_parent_class,
                                     self.contained_class))
        self.max_blocks = max_blocks
        super().__init__(len(layer.linear.weight.chunk(self.max_blocks, 0)))

        # disable exts hooks, buffers only in linear
        layer.linear.disable_exts()

        # convert weight from parameter to buffer
        temp_weight = layer.linear.weight.data
        del layer.linear.weight
        layer.linear.register_buffer('weight', temp_weight)

        # convert bias from parameter to buffer
        temp_bias = None if layer.linear.bias is None\
                else layer.linear.bias.data
        del layer.linear.bias
        layer.linear.register_buffer('bias', temp_bias)

        # register wrapped module
        self._register_wrapped_module(layer)

        chunked_weight = layer.linear.weight.chunk(self.max_blocks, 0)
        chunked_bias = self.num_blocks * [None]\
                if layer.linear.bias is None\
                else layer.linear.bias.chunk(self.max_blocks, 0)

        # checks, TODO: can be removed
        assert self.max_blocks >= self.num_blocks
        assert len(chunked_weight) == self.num_blocks
        assert len(chunked_bias) == self.num_blocks
        # ---

        for idx, (chunk_w, chunk_b) in enumerate(zip(chunked_weight,
                                                     chunked_bias)):
            out_features = chunk_w.size()[0]
            parallel_idx = self.contained_class(
                    in_features=self.main.linear.in_features,
                    out_features=out_features,
                    bias=layer.linear.bias is not None)

            # disable hooks, buffers
            parallel_idx.activation.disable_exts()
            if idx != 0:
                parallel_idx.linear.disable_exts()

            # copy weight
            parallel_idx.linear.weight.data = chunk_w
            # copy bias
            if layer.linear.bias is not None:
                parallel_idx.linear.bias.data = chunk_b
            self.main.add_module(self._parallel_module_name(idx),
                                 parallel_idx)

    # override
    def hbp_hooks(self):
        """Get necessary buffers for main module from children.
        
        Avoid unnecessary copies"""
        self.register_exts_forward_hook(self.reference_mean_input)

    # --- hooks ---
    @staticmethod
    def reference_mean_input(module, input, output):
        """Save reference of mean_input from first child in others.

        Intended use as forward hook.
        Initialize module buffer 'mean_input' in all other children
        and the main layer.
        """
        mean_input = module._get_parallel_module(0).linear.mean_input
        module.main.linear.register_exts_buffer('mean_input', mean_input)
        for idx, mod in enumerate(module.parallel_children()):
            if idx != 0:
                mod.linear.register_exts_buffer('mean_input', mean_input)
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
        # feed through activation
        activation = self.main.activation(input)
        return cat([child.linear(activation) for child
                    in self.parallel_children()], 1)

    def _before_backward_hessian(self):
        self.spread_grad_output()
        self.spread_grad_phi()
        self.spread_gradgrad_phi()

    def spread_grad_output(self):
        """Spread grad_output chunks into parallel activations.

        Initialize module buffer 'grad_output' in all children.
        """
        for mod, grad_out in zip(
                self.parallel_children(),
                self.main.activation.grad_output.chunk(self.max_blocks, 1)):
            mod.activation.register_exts_buffer('grad_output', grad_out)

    def spread_grad_phi(self):
        """Spread grad_phi chunks into parallel activations.

        Initialize module buffer 'grad_phi' in all children.
        """
        for mod, grad_phi in zip(
                self.parallel_children(),
                self.main.activation.grad_phi.chunk(self.max_blocks, 1)):
            mod.activation.register_exts_buffer('grad_phi', grad_phi)

    def spread_gradgrad_phi(self):
        """Spread gradgrad_phi chunks into parallel activations.

        Initialize module buffer 'gradgrad_phi' in all children.
        """
        for mod, gradgrad_phi in zip(
                self.parallel_children(),
                self.main.activation.gradgrad_phi.chunk(self.max_blocks, 1)):
            mod.activation.register_exts_buffer('gradgrad_phi', gradgrad_phi)
