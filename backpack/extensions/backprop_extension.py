"""Implements the backpropagation mechanism."""
import warnings
from typing import Type

import torch.nn
from torch.nn import Sequential

from backpack.extensions.module_extension import ModuleExtension
from backpack.utils.hooks import no_op

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

    def __init__(self, savefield, module_exts, fail_mode=FAIL_ERROR):
        """Initializes parameters.

        Args:
            savefield(str): Where to save results
            module_exts(dict): Maps module classes to `ModuleExtension` instances
            fail_mode(str, optional): Behavior when encountering an unknown layer.
                Can be
                - "ERROR": raise a NotImplementedError
                - "WARN": raise a UserWarning
                - "SILENT": skip the module silently
                Defaults to FAIL_ERROR = "ERROR"
        """
        self.savefield = savefield
        self.__module_extensions = module_exts
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

    def __get_module_extension(self, module):
        module_extension = self.__module_extensions.get(module.__class__)

        if module_extension is None:

            if isinstance(module, Sequential):
                return no_op

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

            return no_op

        return module_extension.apply

    def apply(self, module, g_inp, g_out):
        """Applies backpropagation.

        Args:
            module(torch.nn.module): module to perform backpropagation on
            g_inp(tuple[torch.Tensor]): input gradient
            g_out(tuple[torch.Tensor]): output gradient
        """
        module_extension = self.__get_module_extension(module)
        module_extension(self, module, g_inp, g_out)
