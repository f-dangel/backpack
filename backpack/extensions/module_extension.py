import warnings

from backpack.branching import is_branch_point, is_merge_point


class ModuleExtension:
    """
    Base class for a Module Extension for BackPACK.

    Descendants of this class need to
    - define what parameters of the Module need to be treated (weight, bias)
      and provide functions to compute the quantities
    - extend the `backpropagate` function if information other than the gradient
      needs to be propagated through the graph.
    """

    def __init__(self, params=None):
        """
        Parameters
        ----------
        params: [str]
            List of module parameters that need special treatment.
            for each param `p` in the list, instances of the extended module `m`
            need to have a field `m.p` and the class extending `ModuleExtension`
            need to provide a method with the same signature as the `backprop`
            method.
            The result of this method will be saved in the savefield of `m.p`.
        """
        if params is None:
            params = []

        self.__params = params

        for param in self.__params:
            extFunc = getattr(self, param, None)
            if extFunc is None:
                raise NotImplementedError

    def backpropagate(self, ext, module, g_inp, g_out, bpQuantities):
        """
        Main method to extend to backpropagate additional information through
        the graph.

        Parameters
        ----------
        ext: BackpropExtension
            Instance of the extension currently running
        module: torch.nn.Module
            Instance of the extended module
        g_inp: [Tensor]
            Gradient of the loss w.r.t. the inputs
        g_out: Tensor
            Gradient of the loss w.r.t. the output
        bpQuantities:
            Quantities backpropagated w.r.t. the output

        Returns
        -------
        bpQuantities:
            Quantities backpropagated w.r.t. the input
        """
        warnings.warn("Backpropagate has not been overwritten")

    def apply(self, ext, module, g_inp, g_out):
        """
        Fetch backpropagated quantities from module output, apply backpropagation
        rule, and attach the result to module input(s).
        """
        inp = module.input0
        out = module.output

        bpQuantities = self.__backproped_quantities(ext, out)

        for param in self.__params:
            if self.__param_exists_and_requires_grad(module, param):
                extFunc = getattr(self, param)
                extValue = extFunc(ext, module, g_inp, g_out, bpQuantities)
                self.__save(extValue, ext, module, param)

        bpQuantities = self.backpropagate(ext, module, g_inp, g_out, bpQuantities)

        # input to a merge point is a container of multiple inputs
        module_inputs = inp if is_merge_point(out) else (inp,)

        # distribute backproped quantities to all inputs
        for module_inp in module_inputs:
            self.__backprop_quantities(ext, module_inp, out, bpQuantities)

    @staticmethod
    def __backproped_quantities(ext, out):
        """Fetch backpropagated quantities attached to the module output."""
        return getattr(out, ext.savefield, None)

    @staticmethod
    def __backprop_quantities(ext, inp, out, bpQuantities):
        """Propagate back additional information by attaching it to the module input.

        When the computation graph has branches, multiple quantities will be
        backpropagated to the same input. In this case, a rule for how this information
        should be accumulated must be specified.
        """
        attach = bpQuantities

        existing = getattr(inp, ext.savefield, None)
        should_accumulate = is_branch_point(inp) and existing is not None

        if should_accumulate:
            attach = ext.accumulate_backpropagated_quantities(existing, attach)

        setattr(inp, ext.savefield, attach)

        ModuleExtension.__maybe_delete_received_bpQuantities(ext, inp, out)

    @staticmethod
    def __maybe_delete_received_bpQuantities(ext, inp, out):
        """Delete additional backprop info attached to a module output if possible."""
        is_a_leaf = out.grad_fn is None
        retain_grad_is_on = getattr(out, "retains_grad", False)
        inp_is_out = id(inp) == id(out)

        should_retain_grad = is_a_leaf or retain_grad_is_on or inp_is_out

        if not should_retain_grad:
            if hasattr(out, ext.savefield):
                delattr(out, ext.savefield)

    @staticmethod
    def __param_exists_and_requires_grad(module, param):
        param_exists = getattr(module, param) is not None
        return param_exists and getattr(module, param).requires_grad

    @staticmethod
    def __save(value, extension, module, param):
        setattr(getattr(module, param), extension.savefield, value)


class MergeModuleExtension(ModuleExtension):
    """Handle backpropagation at a merge point. Passes on backpropagated info."""

    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        return backproped
