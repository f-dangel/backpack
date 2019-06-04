"""Parallel series of HBPCompositionActivationLinear."""

from torch.nn.parameter import Parameter
from torch import cat
from ..combined import HBPCompositionActivationLinear
from .parallel import HBPParallel


class HBPParallelCompositionActivationLinear(HBPParallel):
    """Handle HBP for parallel series of HBPCompositionActivationLinear."""
    contained_parent_class = HBPCompositionActivationLinear

    def __init__(self, layer, max_blocks):
        # check class
        self.contained_class = layer.__class__
        if not issubclass(self.contained_class, self.contained_parent_class):
            raise ValueError('Expecting layers derived from {}, got {}'.format(
                self.contained_parent_class, self.contained_class))
        # find out actual number of parallel children
        self.max_blocks = max_blocks
        super().__init__(len(layer.linear.weight.chunk(self.max_blocks, 0)))

        # create parallel children with chunked values of weights/bias
        parallel_children = self._create_parallel_children(layer)

        # register the main layer, transform its parameters to usual buffers
        self.add_module('main', layer)
        for name, child in parallel_children.items():
            self.main.add_module(name, child)
        self._create_main_parameters()

    def _create_main_parameters(self):
        """Remove weight/bias `Parameters` from main module. Concatenate
        weight/bias chunks from parallel children and initialize the
        concatenated tensors as variable buffers. Make parameters of
        children point to location in concatenated versions."""
        # remove weight parameter from main layer, make variable
        del self.main.linear.weight
        self.main.linear.weight = cat(
            [c.linear.weight for c in self.parallel_children()])
        # remove bias parameter from main layer, make variable
        has_bias = self.main.linear.bias is not None
        del self.main.linear.bias
        self.main.linear.bias = None if not has_bias else\
                cat([c.linear.bias for c in self.parallel_children()])
        # point chunked parameters to concatenated location in memory
        w_chunks = self.main.linear.weight.data.chunk(self.max_blocks, 0)
        b_chunks = [None] * self.num_blocks if self.main.linear.bias is None\
                else self.main.linear.bias.data.chunk(self.max_blocks, 0)
        for w, b, child in zip(w_chunks, b_chunks, self.parallel_children()):
            child.linear.weight.data = w
            if child.linear.bias is not None:
                child.linear.bias.data = b

    def _create_parallel_children(self, layer):
        """Create parallel children, copy chunked values of weight/bias.

        Parameters:
        -----------
        layer : (HBPCompositionActivationLinear)
            Composition layer whose weight/bias parameters are to be chunked
            and copied over to the parallel children

        Returns:
        --------
        (dict)
            Dictionary with (key, value) pairs corresponding to the name
            of the child layer and the child layer itself.
        """
        # chunk layer weights/bias
        w_chunks = layer.linear.weight.data.chunk(self.max_blocks, 0)
        b_chunks = [None] * self.num_blocks if layer.linear.bias is None\
                else layer.linear.bias.data.chunk(self.max_blocks, 0)
        out_features = [w.size(0) for w in w_chunks]
        in_features = layer.linear.in_features
        # create parallel children
        children = {}
        for idx, (out, w, b) in enumerate(
                zip(out_features, w_chunks, b_chunks)):
            name = self._parallel_module_name(idx)
            parallel = self.contained_class(
                in_features=in_features, out_features=out, bias=b is not None)
            # disable hooks, buffers
            parallel.disable_exts()
            # set parameters
            parallel.linear.weight = Parameter(w)
            if b is not None:
                parallel.linear.bias = Parameter(b)
            children[name] = parallel
        return children

    # override
    def _apply(self, fn):
        """Need to restore references to chunked weights and bias terms after
        casting to different device/data type."""
        self = super()._apply(fn)
        # restore references
        self._create_main_parameters()
        return self

    # override
    def hbp_hooks(self):
        """Get necessary buffers for main module from children.

        Avoid unnecessary copies"""
        self.register_exts_forward_hook(self.reference_mean_input)

    # --- hooks ---
    @staticmethod
    def reference_mean_input(module, input, output):
        """Save reference of `mean_input` from `main` module in children,
        avoiding copy.

        Intended use as forward hook.
        Initialize module buffer 'mean_input' in all children linear
        submodules.
        """
        mean_input = module.main.linear.mean_input
        for idx, mod in enumerate(module.parallel_children()):
            mod.linear.register_exts_buffer('mean_input', mean_input)

    # --- end of hooks ---

    # override
    def parameter_hessian(self, output_hessian):
        """Split output Hessian into blocks, compute parameter Hessian of
        parallel modules."""
        # split output_hessian
        out_h_split = [(output_hessian.chunk(self.max_blocks, 0)[i]).chunk(
            self.max_blocks, 1)[i] for i in range(self.num_blocks)]
        # call parameter Hessian recursively
        for mod, out_h in zip(self.parallel_children(), out_h_split):
            mod.parameter_hessian(out_h)

    # override
    def forward(self, input):
        """Feed through main layer."""
        # feed through activation
        return self.main(input)

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
