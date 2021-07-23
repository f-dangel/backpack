"""Implements the backpropagation mechanism."""
from __future__ import annotations

import warnings
from typing import Callable, Dict, Tuple, Type

import torch.nn
from torch import Tensor
from torch.nn import Module, Sequential

from backpack.custom_module.branching import Branch, Parallel
from backpack.custom_module.reduce_tuple import ReduceTuple
from backpack.extensions.module_extension import ModuleExtension
from backpack.utils import TORCH_VERSION_AT_LEAST_1_9_0
from backpack.utils.hooks import apply_no_op

if TORCH_VERSION_AT_LEAST_1_9_0:
    from torch.fx import GraphModule

FAIL_ERROR = "ERROR"
FAIL_WARN = "WARN"
FAIL_SILENT = "SILENT"


class BackpropExtension:
    """Base class for the BackPACK extensions.

    Descendants of this class need to
    - define in what field to save results
    - provide a mapping from Module classes to ModuleExtension instances.

    They can then be passed to the Backpack context manager, i.e.,
    ```
    with backpack(NewPackpropExtension("myfield", module_to_extensions)):
        loss(model(X), Y).backward()

        for p in model.parameters():
            print(p.myfield)
    ```
    """

    def __init__(
        self,
        savefield: str,
        module_exts: Dict[Type[Module], ModuleExtension],
        fail_mode: str = FAIL_ERROR,
    ):
        """Initializes parameters.

        Args:
            savefield: Where to save results
            module_exts: Maps module classes to `ModuleExtension` instances
            fail_mode: Behavior when encountering an unknown layer.
                Can be
                - "ERROR": raise a NotImplementedError
                - "WARN": raise a UserWarning
                - "SILENT": skip the module silently
                Defaults to FAIL_ERROR = "ERROR"
        """
        self.savefield = savefield
        self.__module_extensions: Dict[Type[Module], ModuleExtension] = module_exts
        self.__fail_mode = fail_mode

    def set_module_extension(
        self,
        module: Type[torch.nn.Module],
        extension: ModuleExtension,
        overwrite: bool = False,
    ) -> None:
        """Adds a module mapping to module_extensions.

        This can be used to add a custom module.

        Args:
            module: The module that is supposed to be extended
            extension: The custom extension of that module.
            overwrite: Whether to allow overwriting of an existing key.
                Defaults to False.

        Raises:
            ValueError: If the key already exists and overwrite is set to False.
        """
        if overwrite is False and module in self.__module_extensions:
            raise ValueError(
                f"{module} maps to {self.__module_extensions.get(module)}! "
                "Use overwrite = True to force replacement."
            )
        self.__module_extensions[module] = extension

    def __get_module_extension(
        self, module: Module
    ) -> Callable[[BackpropExtension, Module, Tuple[Tensor], Tuple[Tensor]], None]:
        module_extension = self.__module_extensions.get(module.__class__)

        if module_extension is None:

            if isinstance(
                module,
                (GraphModule, Sequential, Branch, Parallel, ReduceTuple)
                if TORCH_VERSION_AT_LEAST_1_9_0
                else (Sequential, Branch, Parallel, ReduceTuple),
            ):
                return apply_no_op

            if self.__fail_mode is FAIL_ERROR:
                raise NotImplementedError(
                    "Extension saving to {} ".format(self.savefield)
                    + "does not have an extension for "
                    + "Module {}".format(module.__class__)
                )
            elif self.__fail_mode == FAIL_WARN:
                warnings.warn(
                    "Extension saving to {} ".format(self.savefield)
                    + "does not have an extension for "
                    + "Module {}".format(module.__class__)
                )

            return apply_no_op

        return module_extension.apply

    def apply(self, module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor]) -> None:
        """Applies backpropagation.

        Args:
            module: module to perform backpropagation on
            g_inp: input gradient
            g_out: output gradient
        """
        module_extension = self.__get_module_extension(module)
        module_extension(self, module, g_inp, g_out)

    # TODO: discuss whether this is necessary or always existing+other
    def accumulate_backpropagated_quantities(self, existing, other):
        """Specify how to accumulate info that is backpropagated to the same node.

        Must be implemented by second-order extensions to function on computation
        graphs with branching.

        For instance, ``DiagGGN`` extensions must sum their quantities, while
        ``curvmatprod`` extensions must accumulate functions to a sum of functions.

        Args:
            existing: already existing backpropagated quantity
            other: new backpropagated quantity

        Raises:
            NotImplementedError: if not overwritten
        """
        raise NotImplementedError(
            f"{self}: No accumulation rule for backpropagated info specified"
        )
