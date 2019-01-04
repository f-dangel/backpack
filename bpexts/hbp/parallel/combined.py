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

    def __init__(self, *layers):
        different_classes = set(l.__class__ for l in layers)
        if not len(different_classes) == 1:
            raise ValueError('Expecting layers of identical type,'
                             ' got {}'.format(different_classes))
        self.contained_class = different_classes.pop()
        if not issubclass(self.contained_class,
                          self.contained_parent_class):
            raise ValueError('Expecting layers derived from {}, got {}'
                             .format(self.contained_parent_class,
                                     self.contained_class))
        super().__init__(*layers)

        self.out_features_list = [c.linear.out_features
                                  for c in self.children()]

        # disable hooks in layers except for first layer
        for i, mod in enumerate(self.children()):
            if i != 0:
                mod.disable_exts()

        # try to copy already existing duplicate buffers from HBP
        try:
            mean_input = layers[0].linear.mean_input
            self.get_submodule(0).linear.register_exts_buffer(
                    'mean_input', mean_input)
            self._reference_mean_input_in_children()
        except AttributeError as e:
            warn('Could not copy/find buffer mean_input.\n{}'.format(e))

        # Try to save memory
        #####################
        has_bias = set(mod.linear.bias is not None
                       for mod in self.children())
        if not len(has_bias) == 1:
            raise ValueError('Expect simultaneous presence/absence'
                             ' of bias, got {}'.format(has_bias))
        has_bias = has_bias.pop()

        # concatenate weight matrix
        weight = cat([mod.linear.weight.data
                      for mod in self.children()])
        self.register_buffer('weight', weight)

        # concatenate bias
        bias = None if not has_bias else\
            cat([mod.linear.bias.data for mod in self.children()])
        self.register_buffer('bias', bias)

        # assign slices to children parameters
        idx = [0] + list(cumsum(self.out_features_list))
        idx = [(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]
        for (i, j), child in zip(idx, self.children()):
            child.linear.weight.data = self.weight.data[i:j, :]
            if has_bias:
                child.linear.bias.data = self.bias.data[i:j]

    # override
    def forward(self, input):
        """Only use first activation layer, split activation output and
        apply each linear module separately."""
        activation_out = self.get_submodule(0).activation(input)
        split_output = [layer.linear(activation_out)
                        for layer in self.children()]
        return self.concatenate_output(split_output)

    # override
    def hbp_hooks(self):
        """Install reference to buffer `mean_input` for all children."""
        self.register_exts_forward_hook(self.reference_mean_input)

    # --- hooks ---
    @staticmethod
    def reference_mean_input(module, input, output):
        """Save reference of mean_input from first child in others.

        Intended use as forward hook.
        Initialize module buffer 'mean_input' in all other linear
        layers of the children beside the first one.
        """
        module._reference_mean_input_in_children()

    def _reference_mean_input_in_children(self):
        """Store a reference to the buffer mean_input in each child.

        Avoid copies of the same tensor.
        """
        mean_input = self.get_submodule(0).linear.mean_input
        for i, mod in enumerate(self.children()):
            if i != 0:
                mod.linear.register_exts_buffer('mean_input', mean_input)
    # --- end of hooks ---

    def unite(self):
        """Unite all parallel children to a single one.

        Returns:
        --------
        (HBPParallelCompositionActivation)
            Parallel series of HBPCompositionActivationLinear consisting of
            a single child, behaves identically in forward mode.
        """
        out_features = sum(c.linear.out_features for c in self.children())
        in_features = set(c.linear.in_features for c in self.children())

        # check consistency
        if not len(in_features) == 1:
            raise ValueError('Expect same in_features, got {}'
                             .format(in_features))
        in_features = in_features.pop()

        # check consistency
        has_bias = set(c.linear.bias is not None for c in self.children())
        if not len(has_bias) == 1:
            raise ValueError('Expect simultaneous presence/absence'
                             ' of bias, got {}'.format(has_bias))
        has_bias = has_bias.pop()

        layer = self.contained_class(in_features=in_features,
                                     out_features=out_features,
                                     bias=has_bias)
        layer.linear.weight.data = self.weight
        # concatenate bias and assign
        if has_bias:
            layer.linear.bias.data = self.bias.data

        # try to copy mean_input from linear
        try:
            mean_input = self.get_submodule(0).linear.mean_input
            layer.linear.register_exts_buffer('mean_input',
                                              mean_input)
        except AttributeError:
            warn('Could not copy/find buffer mean_input')

        # copy buffers from activation
        ##############################
        # buffer grad_output
        try:
            grad_output = self.get_submodule(0).activation.grad_output
            layer.activation.register_exts_buffer('grad_output',
                                                  grad_output)
        except AttributeError:
            warn('Could not copy/find buffer grad_output')

        # buffer grad_phi
        try:
            grad_phi = self.get_submodule(0).activation.grad_phi
            layer.activation.register_exts_buffer('grad_phi',
                                                  grad_phi)
        except AttributeError:
            warn('Could not copy/find buffer grad_phi')

        # buffer gradgrad_phi
        try:
            gradgrad_phi = self.get_submodule(0).activation.gradgrad_phi
            layer.activation.register_exts_buffer('gradgrad_phi',
                                                  gradgrad_phi)
        except AttributeError:
            warn('Could not copy/find buffer gradgrad_phi')

        return self.__class__(layer)

    def split(self, out_features_list):
        """Split into parallel series of HBPCompositionActivationLinear.

        Parameters:
        -----------
        out_features_list : (list(int))
            Output features for each of the parallel modules

        Returns:
        --------
        (HBPParallelCompositionActivationLinear)
        """
        united = self.unite()

        # check consistency
        if not sum(out_features_list) == sum(united.out_features_list):
            raise ValueError('Invalid splitting: {} does not sum'
                             'to {}'.format(out_features_list,
                                            united.out_features_list))

        # get the single HBPCompositionActivationLinear child
        combined = united.get_submodule(0)
        in_features = combined.linear.in_features
        has_bias = (combined.linear.bias is not None)

        # create parallel children
        layers = []

        idx = [0] + list(cumsum(out_features_list))
        idx = [(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]
        for out, (i, j) in zip(out_features_list, idx):
            # create HBPCompositionActivationLinear
            child = self.contained_class(in_features=in_features,
                                         out_features=out,
                                         bias=has_bias)
            child.linear.weight.data = united.weight.data[i:j, :]
            if has_bias:
                child.linear.bias.data = united.bias.data[i:j]
            layers.append(child)


        # register buffers grad_output, grad_phi, gradgrad_phi
        # buffer grad_output
        activation = combined.activation
        try:
            grad_output = activation.grad_output
            layers[0].activation.register_exts_buffer('grad_output',
                                                      grad_output)
        except AttributeError:
            warn('Could not copy/find buffer grad_output')

        # buffer grad_phi
        try:
            grad_phi = activation.grad_phi
            layers[0].activation.register_exts_buffer('grad_phi',
                                                      grad_phi)
        except AttributeError:
            warn('Could not copy/find buffer grad_phi')

        # buffer gradgrad_phi
        try:
            gradgrad_phi = activation.gradgrad_phi
            layers[0].activation.register_exts_buffer('gradgrad_phi',
                                                      gradgrad_phi)
        except AttributeError:
            warn('Could not copy/find buffer gradgrad_phi')

        # try to copy mean_input from linear
        try:
            mean_input = self.get_submodule(0).linear.mean_input
            layers[0].linear.register_exts_buffer('mean_input',
                                                  mean_input)
        except AttributeError:
            warn('Could not copy/find buffer mean_input')

        return self.__class__(*layers)
