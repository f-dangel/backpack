"""Contains base class for BackPACK module extensions."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, List, Tuple

from torch import Tensor
from torch.nn import Flatten, Module

from backpack.utils import TORCH_VERSION_AT_LEAST_1_9_0
from backpack.utils.module_classification import is_loss

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
                For each param `p` in the list, instances of the extended module `m`
                need to have a field `m.p` and the class extending `ModuleExtension`
                needs to provide a method with the same signature as the `backpropagate`
                method.
                The result of this method will be saved in the savefield of `m.p`.

        Raises:
            NotImplementedError: if child class doesn't have a method for each parameter
        """
        self.__params: List[str] = [] if params is None else params

        for param in self.__params:
            if not hasattr(self, param):
                raise NotImplementedError(
                    f"The module extension {self} is missing an implementation "
                    f"of how to calculate the quantity for {param}. "
                    f"This should be realized in a function "
                    f"{param}(extension, module, g_inp, g_out, bpQuantities) -> Any."
                )

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

    def __call__(
        self,
        extension: BackpropExtension,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
    ) -> None:
        """Apply all actions required by the extension.

        Fetch backpropagated quantities from module output, apply backpropagation
        rule, and attach the result to module input(s).

        Args:
            extension: current backpropagation extension
            module: current module
            g_inp: input gradients
            g_out: output gradients

        Raises:
            AssertionError: if there is no saved quantity although extension expects one
        """
        delete_old_quantities = not self.__should_retain_backproped_quantities(module)
        bp_quantity = self.__get_backproped_quantity(
            extension, module.output, delete_old_quantities
        )
        if (
            extension.expects_backpropagation_quantities()
            and bp_quantity is None
            and not is_loss(module)
        ):
            if not TORCH_VERSION_AT_LEAST_1_9_0 and isinstance(module, Flatten):
                return
            else:
                raise AssertionError(
                    "BackPACK extension expects a backpropagation quantity but it is None. "
                    f"Module: {module}, Extension: {extension}."
                )

        for param in self.__params:
            if self.__param_exists_and_requires_grad(module, param):
                extFunc = getattr(self, param)
                extValue = extFunc(extension, module, g_inp, g_out, bp_quantity)
                self.__save_value_on_parameter(extValue, extension, module, param)

        if self.__should_backpropagate(extension, module):
            bp_quantity = self.backpropagate(
                extension, module, g_inp, g_out, bp_quantity
            )
            self.__save_backproped_quantity(extension, module.input0, bp_quantity)

    @staticmethod
    def __should_backpropagate(extension: BackpropExtension, module: Module) -> bool:
        """Determines whether the current extension should perform a backpropagation.

        Args:
            extension: current extension
            module: current module

        Returns:
            whether a backpropagation should be performed
        """
        input_requires_grad: bool = module.input0.requires_grad
        return input_requires_grad and extension.expects_backpropagation_quantities()

    @staticmethod
    def __should_retain_backproped_quantities(module: Module) -> bool:
        """Whether the backpropagation quantities should be kept.

        This is old code inherited and not tested.

        Args:
            module: current module

        Returns:
            whether backpropagation quantities should be kept
        """
        is_a_leaf = module.output.grad_fn is None
        retain_grad_is_on = getattr(module.output, "retains_grad", False)
        # inp_is_out = id(module.input0) == id(module.output)
        should_retain_grad = is_a_leaf or retain_grad_is_on  # or inp_is_out
        return should_retain_grad

    @staticmethod
    def __get_backproped_quantity(
        extension: BackpropExtension,
        reference_tensor: Tensor,
        delete_old: bool,
    ) -> Tensor or None:
        """Fetch backpropagated quantities attached to the module output.

        The property reference_tensor.data_ptr() is used as a reference.

        Args:
            extension: current BackPACK extension
            reference_tensor: the output Tensor of the current module
            delete_old: whether to delete the old backpropagated quantity

        Returns:
            the backpropagation quantity
        """
        return extension.saved_quantities.retrieve_quantity(
            reference_tensor.data_ptr(), delete_old
        )

    @staticmethod
    def __save_backproped_quantity(
        extension: BackpropExtension, reference_tensor: Tensor, bpQuantities: Any
    ) -> None:
        """Save additional information backpropagated for a tensor.

        Args:
            extension: current BackPACK extension
            reference_tensor: reference tensor for which additional information
                is backpropagated.
            bpQuantities: backpropagation quantities that should be saved
        """
        extension.saved_quantities.save_quantity(
            reference_tensor.data_ptr(), bpQuantities
        )

    @staticmethod
    def __param_exists_and_requires_grad(module: Module, param_str: str) -> bool:
        """Whether the module has the parameter and it requires gradient.

        Args:
            module: current module
            param_str: parameter name

        Returns:
            whether the module has the parameter and it requires gradient
        """
        param_exists = getattr(module, param_str) is not None
        return param_exists and getattr(module, param_str).requires_grad

    @staticmethod
    def __save_value_on_parameter(
        value: Any, extension: BackpropExtension, module: Module, param_str: str
    ) -> None:
        """Saves the value on the parameter of that module.

        Args:
            value: The value that should be saved.
            extension: The current BackPACK extension.
            module: current module
            param_str: parameter name
        """
        setattr(getattr(module, param_str), extension.savefield, value)
