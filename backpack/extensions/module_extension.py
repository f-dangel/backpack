"""Contains base class for extending torch.nn.Module."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, List, Tuple, Union

from torch import Tensor
from torch.nn import Module

from backpack.custom_module.branching import is_branch_point, is_merge_point

if TYPE_CHECKING:
    from backpack import BackpropExtension


class ModuleExtension:
    """Base class for a Module Extension for BackPACK.

    Descendants of this class need to
    - define what parameters of the Module need to be treated (weight, bias)
      and provide functions to compute the quantities
    - extend the `backpropagate` function if information other than the gradient
      needs to be propagated through the graph.
    """

    def __init__(self, params: List[str] = None):
        """Initialization.

        Args:
            params: List of module parameters that need special treatment.
                for each param `p` in the list, instances of the extended module `m`
                need to have a field `m.p` and the class extending `ModuleExtension`
                need to provide a method with the same signature as the `backprop`
                method.
                The result of this method will be saved in the savefield of `m.p`.

        Raises:
            NotImplementedError: if for one of the parameters, the function named
                after the parameter does not exist
        """
        if params is None:
            params = []

        self.__params: List[str] = params

        for param in self.__params:
            extFunc = getattr(self, param, None)
            if extFunc is None:
                raise NotImplementedError

    def backpropagate(
        self,
        ext: BackpropExtension,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        bpQuantities: Any,
    ) -> Any:
        """Backpropagate additional information through the graph.

        Args:
            ext: Instance of the extension currently running
            module: Instance of the extended module
            g_inp: Gradient of the loss w.r.t. the inputs
            g_out: Gradient of the loss w.r.t. the output
            bpQuantities: Quantities backpropagated w.r.t. the output

        # noqa: DAR202
        Returns:
            Quantities backpropagated w.r.t. the input
        """
        warnings.warn("Backpropagate has not been overwritten")

    def apply(
        self,
        ext: BackpropExtension,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
    ) -> None:
        """Apply module extension operations.

        Fetch backpropagated quantities from module output, apply backpropagation
        rule, and attach the result to module input(s).

        Args:
            ext: extension currently running
            module: extended module
            g_inp: input gradients
            g_out: output gradients
        """
        inp: Union[Tuple[Tensor], Tensor] = module.input0
        out: Tensor = module.output

        bpQuantities = self.__backproped_quantities(ext, out)

        for param in self.__params:
            if self.__param_exists_and_requires_grad(module, param):
                extFunc = getattr(self, param)
                extValue = extFunc(ext, module, g_inp, g_out, bpQuantities)
                self.__save(extValue, ext, module, param)

        bpQuantities = self.backpropagate(ext, module, g_inp, g_out, bpQuantities)

        # input to a merge point is a container of multiple inputs
        # TODO make this more general
        module_inputs: Tuple[Tensor] = (
            (module.input0, module.input1) if is_merge_point(out) else (inp,)
        )

        # distribute backproped quantities to all inputs
        for module_inp in module_inputs:
            self.__backprop_quantities(ext, module_inp, out, bpQuantities)

    @staticmethod
    def __backproped_quantities(ext: BackpropExtension, out: Tensor) -> Any:
        """Fetch backpropagated quantities attached to the module output.

        Args:
            ext: extension currently running
            out: output tensor of current module

        Returns:
            saved quantity from backpropagation, retrieved as property of out
        """
        return getattr(out, ext.savefield, None)

    @staticmethod
    def __backprop_quantities(
        ext: BackpropExtension, inp: Tensor, out: Tensor, bpQuantities: Any
    ) -> None:
        """Propagate back additional information by attaching it to the module input.

        When the computation graph has branches, multiple quantities will be
        backpropagated to the same input. In this case, a rule for how this information
        should be accumulated must be specified.

        Args:
            ext: extension currently running
            inp: input0 of current module
            out: output of current module
            bpQuantities: backpropagation quantities
        """
        attach = bpQuantities

        # is True for branch points
        if hasattr(inp, ext.savefield):
            setattr(inp, ext.savefield, getattr(inp, ext.savefield) + attach)
        else:
            setattr(inp, ext.savefield, attach)

        ModuleExtension.__maybe_delete_received_bpQuantities(ext, inp, out)

    @staticmethod
    def __maybe_delete_received_bpQuantities(
        ext: BackpropExtension, inp: Tensor, out: Tensor
    ) -> None:
        """Delete additional backprop info attached to a module output if possible.

        Args:
            ext: extension currently running
            inp: input0 of current module
            out: output of current module
        """
        is_a_leaf = out.grad_fn is None
        retain_grad_is_on = getattr(out, "retains_grad", False)
        inp_is_out = id(inp) == id(out)

        should_retain_grad = is_a_leaf or retain_grad_is_on or inp_is_out

        if not should_retain_grad:
            if hasattr(out, ext.savefield):
                delattr(out, ext.savefield)

    @staticmethod
    def __param_exists_and_requires_grad(module: Module, param: str) -> bool:
        """Determines whether parameter exists and requires gradient.

        Args:
            module: module
            param: parameter name

        Returns:
            whether param exists and requires grad
        """
        param_exists = getattr(module, param) is not None
        return param_exists and getattr(module, param).requires_grad

    @staticmethod
    def __save(
        value: Any, extension: BackpropExtension, module: Module, param: str
    ) -> None:
        """Save some value on the parameter of the module.

        Args:
            value: The value to be saved
            extension: The current extension, determines save_field name
            module: the module which has the parameter
            param: parameter name, on this param the value is saved
        """
        setattr(getattr(module, param), extension.savefield, value)


class MergeModuleExtension(ModuleExtension):
    """Handle backpropagation at a merge point. Passes on backpropagated info."""

    def backpropagate(
        self,
        ext: BackpropExtension,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        bpQuantities: Any,
    ) -> Any:
        """Backpropagate additional information through the graph.

        Args:
            ext: Instance of the extension currently running
            module: Instance of the extended module
            g_inp: Gradient of the loss w.r.t. the inputs
            g_out: Gradient of the loss w.r.t. the output
            bpQuantities: Quantities backpropagated w.r.t. the output

        Returns:
            Quantities backpropagated w.r.t. the input
        """
        return bpQuantities
