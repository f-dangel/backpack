import warnings

from torch.nn import Sequential

FAIL_ERROR = "ERROR"
FAIL_WARN = "WARN"
FAIL_SILENT = "SILENT"


class BackpropExtension:
    """
    Base class for the BackPACK extensions.

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

    __external_module_extensions = {}

    def __init__(self, savefield, module_exts, fail_mode=FAIL_ERROR):
        """
        Parameters
        ----------
        savefield : str
            Where to save results
        module_exts : dict
            Dictionary mapping module classes to `ModuleExtension` instances
        fail_mode : str, optional
            Behavior when encountering an unknown layer.
            Can be
            - "ERROR": raise a NotImplementedError
            - "WARN": raise a UserWarning
            - "SILENT": skip the module silently
        """
        self.savefield = savefield
        self.__module_extensions = {
            **module_exts,
            **self.__class__.__external_module_extensions,
        }

        self.__fail_mode = fail_mode

    @classmethod
    def add_module_extension(cls, module, extension):
        cls.__external_module_extensions[module] = extension

    def __get_module_extension(self, module):
        module_extension = self.__module_extensions.get(module.__class__)

        def no_op(*args):
            return None

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
        module_extension = self.__get_module_extension(module)
        module_extension(self, module, g_inp, g_out)
