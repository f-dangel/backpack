from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backpack import BackpropExtension


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

    def apply(self, ext: BackpropExtension, module, g_inp, g_out):
        # inp = module.input0
        # out = module.output
        print("ModuleExtension::apply, module           ", type(module))
        for i in range(len(g_out)):
            print(f"ModuleExtension::apply, g_out[{i}]         ", id(g_out[i]))
            print(f"ModuleExtension::apply, g_out[{i}]         ", g_out[i].sum().item())
        for i in range(len(g_inp)):
            print(f"ModuleExtension::apply, g_inp[{i}]         ", id(g_inp[i]))
            print(f"ModuleExtension::apply, g_inp[{i}]         ", g_inp[i].sum().item())
        # bpQuantities = self.__backproped_quantities(ext, out)
        # bpQuantities = ext.saved_quantities.retrieve_quantity(id(g_out[0]))
        bpQuantities = ext.saved_quantities.retrieve_quantity(g_out[0].sum().item())
        # if bpQuantities is None:
        #    bpQuantities = ext.saved_quantities.retrieve_quantity(id(g_inp[0]))
        print("ModuleExtension::apply, bpQuantities     ", type(bpQuantities))

        for param in self.__params:
            if self.__param_exists_and_requires_grad(module, param):
                extFunc = getattr(self, param)
                extValue = extFunc(ext, module, g_inp, g_out, bpQuantities)
                self.__save(extValue, ext, module, param)

        bpQuantities = self.backpropagate(ext, module, g_inp, g_out, bpQuantities)

        # self.__backprop_quantities(ext, inp, out, bpQuantities)
        # ext.saved_quantities.save_quantity(id(g_inp[0]), bpQuantities)
        ext.saved_quantities.save_quantity(g_inp[0].sum().item(), bpQuantities)

    @staticmethod
    def __backproped_quantities(ext, out):
        return getattr(out, ext.savefield, None)

    @staticmethod
    def __backprop_quantities(ext, inp, out, bpQuantities):

        setattr(inp, ext.savefield, bpQuantities)

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
