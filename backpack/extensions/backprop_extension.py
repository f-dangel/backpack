"""Implements the backpropagation mechanism."""
from __future__ import annotations

import abc
import warnings
from abc import ABC
from typing import Dict, Tuple, Type

from torch import Tensor
from torch.nn import Module

from backpack.extensions.module_extension import ModuleExtension
from backpack.extensions.saved_quantities import SavedQuantities

FAIL_ERROR = "ERROR"
FAIL_WARN = "WARNING"
FAIL_SILENT = "SILENT"


class BackpropExtension(ABC):
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

        Raises:
            AssertionError: if fail_mode is not valid
        """
        if fail_mode not in (FAIL_WARN, FAIL_ERROR, FAIL_SILENT):
            raise AssertionError(f"no valid fail mode: {fail_mode}")
        self.saved_quantities: SavedQuantities = SavedQuantities()
        self.savefield: str = savefield
        self.__module_extensions: Dict[Type[Module], ModuleExtension] = module_exts
        self.__fail_mode: str = fail_mode

    def set_module_extension(
        self,
        module: Type[Module],
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

    def __get_module_extension(self, module: Module) -> ModuleExtension or None:
        module_extension = self.__module_extensions.get(module.__class__)

        if module_extension is None:
            if self.__fail_mode is FAIL_ERROR:
                raise NotImplementedError(
                    f"Extension saving to {self.savefield} "
                    "does not have an extension for "
                    f"Module {module.__class__}"
                )
            elif self.__fail_mode == FAIL_WARN:
                for _ in module.parameters():
                    warnings.warn(
                        f"Extension saving to {self.savefield} does not have an "
                        f"extension for Module {module.__class__} "
                        f"although the module has parameters"
                    )
                    break

        return module_extension

    def __call__(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
    ) -> None:
        """Applies backpropagation.

        Args:
            module: module to perform backpropagation on
            g_inp: input gradient
            g_out: output gradient
        """
        module_extension = self.__get_module_extension(module)
        if module_extension is not None:
            module_extension(self, module, g_inp, g_out)

    @abc.abstractmethod
    def expects_backpropagation_quantities(self) -> bool:
        """Whether the extension uses additional backpropagation quantities.

        Returns:
            Whether the extension uses additional backpropagation quantities.
        """
        return

    def clear(self) -> None:
        """Clears backpropagation quantities."""
        self.saved_quantities.clear()
