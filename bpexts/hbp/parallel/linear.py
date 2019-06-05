"""Parallel series of linear layers."""

from torch.nn.parameter import Parameter
from torch import cat
from ..linear import HBPLinear
from .parallel import HBPParallel


class HBPParallelLinear(HBPParallel):
    """Handle backpropagation for a parallel series of linear layers.

    This is done by storing a bunch of so called parallel children layers.
    These layers contain the parameters that are optimized, but for the
    forward pass performance suffers a lot when the input is fed through
    each parallel child sequentially and the output is concatenated into
    the output vector.

    To overcome this, this layer has a submodule called `main`. It's job
    is to use the concatenated versions of the weights and bias terms
    in the forward pass and to make sure that quantities required for
    HBP are correctly referenced in the children modules (otherwhise there
    would be copies of the same quantities floating around, blowing up
    memory consumption).

    To be honest, the implementation is not very clean and requires
    a lot of pointers (corresponding to the chunked parameters) that
    have to point to the correct location of the concatenated parameters.

    There were also some problems caused by PyTorch internals when the
    module is loaded to a device or cast to a different type.
    Consider this implementation to be very likely to break when being
    modified. One should always check whether the tests are still running.
    """
    contained_class = HBPLinear

    # TODO: The approximation mode of the children has to be synced
    # with the main module. Will throw exceptions if changing the approximation

    def __init__(self, layer, max_blocks):
        # check class
        if not layer.__class__ == self.contained_class:
            raise ValueError('Expecting layer of type {}, got {}'.format(
                self.contained_class, layer.__class__))
        # find out actual number of parallel children
        self.max_blocks = max_blocks
        super().__init__(len(layer.weight.chunk(self.max_blocks, 0)))

        # create parallel children with chunked values of weight/bias
        parallel_children = self._create_parallel_children(layer)

        # register the main layer, transform its parameters to usual
        # buffer variables so they do not show up when calling
        # `self.parameters()`
        self.add_module('main', layer)
        for name, child in parallel_children.items():
            self.main.add_module(name, child)
        self._create_main_parameters()

    # override
    def enable_hbp(self):
        # do not enable HBP of children
        super().enable_hbp()
        # try to enable main layer HBP
        try:
            self.main.enable_hbp()
        except AttributeError:
            pass

    # override
    def set_hbp_approximation(self,
                              average_input_jacobian=True,
                              average_parameter_jacobian=True):
        """Not sure if useful to implement"""
        if average_parameter_jacobian is not True:
            raise NotImplementedError
        # very dirty workaround as the approximation modes of this
        # layer will never be different
        self.average_input_jac = None
        self.average_param_jac = average_parameter_jacobian

    def _create_main_parameters(self):
        """Remove weight/bias `Parameters` from main module. Concatenate
        weight/bias chunks from parallel children and initialize the
        concatenated tensors as variable buffers. Make parameters of
        children point to location in concatenated versions."""
        # remove weight parameter from main layer, make variable
        del self.main.weight
        self.main.weight = cat([c.weight for c in self.parallel_children()])
        # remove bias parameter from main layer, make variable
        has_bias = self.main.bias is not None
        del self.main.bias
        self.main.bias = None if not has_bias else\
                cat([c.bias for c in self.parallel_children()])
        # point chunked parameters to concatenated location in memory
        w_chunks = self.main.weight.data.chunk(self.max_blocks, 0)
        b_chunks = [None] * self.num_blocks if self.main.bias is None\
                else self.main.bias.data.chunk(self.max_blocks, 0)
        for w, b, child in zip(w_chunks, b_chunks, self.parallel_children()):
            child.weight.data = w
            if child.bias is not None:
                child.bias.data = b

    def _create_parallel_children(self, layer):
        """Create parallel children, copy chunked values of weight/bias.

        Parameters:
        -----------
        layer : (HBPLinear)
            Linear layer whose weight/bias parameters are to be chunked
            and copied over to the parallel children

        Returns:
        --------
        (dict)
            Dictionary with (key, value) pairs corresponding to the name
            of the child layer and the child layer itself.
        """
        # chunk layer weights/bias
        w_chunks = layer.weight.data.chunk(self.max_blocks, 0)
        b_chunks = [None] * self.num_blocks if layer.bias is None\
                else layer.bias.data.chunk(self.max_blocks, 0)
        out_features = [w.size(0) for w in w_chunks]
        in_features = layer.in_features
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
            parallel.weight = Parameter(w)
            if b is not None:
                parallel.bias = Parameter(b)
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
        """Remove input hook in children, use a single copy instead."""
        if self.average_param_jac == True:
            self.register_exts_forward_hook(self.reference_mean_input)
        elif self.average_param_jac == False:
            self.register_exts_forward_hook(self.reference_input_kron_mean)
        else:
            raise ValueError('Unknown value for average_param_jac : {}'.format(
                self.average_param_jac))

    # --- hooks ---
    @staticmethod
    def reference_mean_input(module, input, output):
        """Save reference of `mean_input` from `main` module in children,
        avoiding copy.

        Intended use as forward hook.
        Initialize module buffer 'mean_input' in all children
        """
        mean_input = module.main.mean_input
        for idx, mod in enumerate(module.parallel_children()):
            mod.register_exts_buffer('mean_input', mean_input)

    @staticmethod
    def reference_input_kron_mean(module, input, output):
        """Save reference of `input_kron_mean` from `main` module in children,
        avoiding copy.

        Intended use as forward hook.
        Initialize module buffer 'input_kron_mean' in all children
        """
        input_kron_mean = module.main.input_kron_mean
        for idx, mod in enumerate(module.parallel_children()):
            mod.register_exts_buffer('input_kron_mean', input_kron_mean)

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
        return self.main(input)
