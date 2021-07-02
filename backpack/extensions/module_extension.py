"""Contains base class for BackPACK module extensions."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, List, Tuple

from torch import Tensor
from torch.nn import Module

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
            NotImplementedError: if child class doesn't have a method for each parameter
        """
        if params is None:
            params = []

        self.__params = params

        for param in self.__params:
            if hasattr(self, param) is False:
                raise NotImplementedError

    def backpropagate(
        self,
        extension: BackpropExtension,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        bpQuantities: Any,
    ) -> Any:
        """Backpropagation of additional information through the graph.

        Args:
            extension: Instance of the extension currently running
            module: Instance of the extended module
            g_inp: Gradient of the loss w.r.t. the inputs
            g_out: Gradient of the loss w.r.t. the output
            bpQuantities: Quantities backpropagated w.r.t. the output

        Returns
            Quantities backpropagated w.r.t. the input
        """
        warnings.warn("Backpropagate has not been overwritten")

    def apply(
        self,
        extension: BackpropExtension,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        use_legacy: bool = False,
    ) -> None:
        """Apply all actions required by the extension.

        Fetch backpropagated quantities from module output, apply backpropagation
        rule, and attach the result to module input(s).

        Args:
            extension: current backpropagation extension
            module: current module
            g_inp: input gradients
            g_out: output gradients
            use_legacy: whether to use the legacy backward hook.
                Deprecated since torch version 1.8.0. Default: False.
        """
        bpQuantities = self.__get_backproped_quantity(
            extension, module.output if use_legacy else g_out[0], use_legacy
        )

        for param in self.__params:
            if self.__param_exists_and_requires_grad(module, param):
                extFunc = getattr(self, param)
                extValue = extFunc(extension, module, g_inp, g_out, bpQuantities)
                self.__save_value_on_parameter(extValue, extension, module, param)

        bpQuantities = self.backpropagate(extension, module, g_inp, g_out, bpQuantities)

        self.__save_backprop_quantity(
            extension,
            module.input0,
            module.output,
            module.input0 if use_legacy else g_inp[0],
            bpQuantities,
            use_legacy,
        )

    @staticmethod
    def __get_backproped_quantity(
        extension: BackpropExtension, reference_tensor: Tensor, use_legacy: bool
    ) -> Tensor or None:
        """Fetch backpropagated quantities attached to the module output.

        Args:
            extension: current BackPACK extension
            reference_tensor: the output Tensor of the current module
            use_legacy: whether to use the legacy backward hook.
                Deprecated since torch version 1.8.0. Default: False.

        Returns:
            the backpropagation quantity
        """
        if use_legacy:
            return getattr(reference_tensor, extension.savefield, None)
        else:
            return extension.saved_quantities.retrieve_quantity(
                reference_tensor.data_ptr()
            )

    @staticmethod
    def __save_backprop_quantity(
        extension: BackpropExtension,
        inp: Tensor,
        out: Tensor,
        reference_tensor: Tensor,
        bpQuantities: Any,
        use_legacy: bool,
    ) -> None:
        """Propagate back additional information by attaching it to the module input.

        Args:
            extension: current BackPACK extension
            inp: input tensor
            out: output tensor
            reference_tensor: reference tensor on which to save
            bpQuantities: backpropagation quantities that should be saved
            use_legacy: whether to use the legacy backward hook.
                Deprecated since torch version 1.8.0. Default: False.
        """
        if use_legacy:
            setattr(reference_tensor, extension.savefield, bpQuantities)

            is_a_leaf = out.grad_fn is None
            retain_grad_is_on = getattr(out, "retains_grad", False)
            inp_is_out = id(inp) == id(out)
            should_retain_grad = is_a_leaf or retain_grad_is_on or inp_is_out

            if not should_retain_grad:
                if hasattr(out, extension.savefield):
                    delattr(out, extension.savefield)
        else:
            extension.saved_quantities.save_quantity(
                reference_tensor.data_ptr(), bpQuantities
            )

    @staticmethod
    def __param_exists_and_requires_grad(module: Module, param: str) -> bool:
        """Whether the module has the parameter and it requires gradient.

        Args:
            module: current module
            param: parameter name

        Returns:
            whether the module has the parameter and it requires gradient
        """
        param_exists = getattr(module, param) is not None
        return param_exists and getattr(module, param).requires_grad

    @staticmethod
    def __save_value_on_parameter(
        value: Any, extension: BackpropExtension, module: Module, param: str
    ) -> None:
        """Saves the value on the parameter of that module.

        Args:
            value: The value that should be saved.
            extension: The current BackPACK extension.
            module: current module
            param: parameter name
        """
        setattr(getattr(module, param), extension.savefield, value)
