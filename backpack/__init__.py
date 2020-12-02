"""BackPACK."""
import inspect

import torch

from backpack.extensions.backprop_extension import BackpropExtension

from . import extensions
from .context import CTX


class backpack:
    """Activates Backpack Extensions.

    Activates the BackPACK extensions passed as arguments for the
    :code:`backward` calls in the current :code:`with` block.
    """

    def __init__(self, *exts: BackpropExtension, debug=False):
        """Activate the Backpack extensions.

        Example usage:
        ```
        X, Y, model, lossfunc = get_problem()

        backpack.extend(model)
        backpack.extend(lossfunc)

        with backpack.backpack(backpack.extensions.Variance()):
            lossfunc(model(X), Y).backward()

            for p in model.parameters():
                print(p.grad)
                print(p.variance)
        ```

        .. warning ::

            The quantities computed by backPACK may be garbage collected when
            exiting the `with` clause. Use them within the `with` clause or
            assign them to another variable.

        Attributes:
            args: [BackpropExtension]
                The extensions to activate for the backward pass.
            debug: Bool, optional (default: False)
                If true, will print debug messages during the backward pass.
        """
        for ext in exts:
            if not isinstance(ext, BackpropExtension):
                if inspect.isclass(ext) and issubclass(ext, BackpropExtension):
                    raise ValueError(
                        "backpack expect instances of BackpropExtension,"
                        + " but received a class instead [{}].".format(ext)
                        + " Instantiate it before passing it to backpack."
                    )
                else:
                    raise ValueError(
                        "backpack expects instances of BackpropExtension,"
                        + " but received [{}].".format(ext)
                    )

        self.exts = exts
        self.debug = debug

    def __enter__(self):
        self.old_CTX = CTX.get_active_exts()
        self.old_debug = CTX.get_debug()
        CTX.set_active_exts(self.exts)
        CTX.set_debug(self.debug)

    def __exit__(self, type, value, traceback):
        CTX.set_active_exts(self.old_CTX)
        CTX.set_debug(self.old_debug)


class disable:
    """Entirely disables BackPACK, including storage of input and output.

    To compute the additional quantities, BackPACK needs to know the input and
    output of the modules in the computation graph. It saves those by default.
    ``disable`` tells BackPACK to _not_ save this information during the forward.

    This can be useful if you only want a gradient with pytorch on a module
    that is ``extended`` with BackPACK and need to avoid memory overhead.
    If you do not need any gradient, use the ``torch.no_grad`` context instead.

    This context is not the exact opposite of the ``backpack`` context.
    The ``backpack`` context enables specific extensions during a backward.
    This context disables storing input/output information during a forward.

    Note:
        ``with backpack(...)`` in a ``with disable()`` context will fail
        even if the forward pass is carried out in ``with backpack(...)``.
    """

    store_io = True

    def __enter__(self):
        """Disable input/output storing."""
        self.old_store_io = disable.store_io
        disable.store_io = False

    def __exit__(self, type, value, traceback):
        """Set input/output storing to old value."""
        disable.store_io = self.old_store_io

    @staticmethod
    def should_store_io():
        """Return whether input and output should be stored."""
        return disable.store_io


def hook_store_io(module, input, output):
    """Saves the input and output as attributes of the module.

    Args:
        module: module
        input: List of input tensors
        output: output tensor
    """
    if disable.should_store_io() and torch.is_grad_enabled():
        for i in range(len(input)):
            setattr(module, "input{}".format(i), input[i])
            setattr(
                module, "input{}_shape".format(i), torch.IntTensor([*input[i].size()])
            )
        setattr(module, "output", output)
        setattr(module, "output_shape", torch.IntTensor([*output.size()]))


def memory_cleanup(module):
    """Remove I/O stored by backpack during the forward pass.

    Deletes the attributes created by `hook_store_io` and `hook_store_shapes`.
    """
    if hasattr(module, "output"):
        delattr(module, "output")
    if hasattr(module, "output_shape"):
        delattr(module, "output_shape")
    i = 0
    while hasattr(module, "input{}".format(i)):
        delattr(module, "input{}".format(i))
        i += 1
    i = 0
    while hasattr(module, "input{}_shape".format(i)):
        delattr(module, "input{}_shape".format(i))
        i += 1


def hook_run_extensions(module, g_inp, g_out):
    for backpack_extension in CTX.get_active_exts():
        if CTX.get_debug():
            print("[DEBUG] Running extension", backpack_extension, "on", module)
        backpack_extension.apply(module, g_inp, g_out)

    if not (
        CTX.is_extension_active(
            extensions.curvmatprod.HMP,
            extensions.curvmatprod.GGNMP,
            extensions.curvmatprod.PCHMP,
        )
    ):
        memory_cleanup(module)


def extend(module: torch.nn.Module, debug=False):
    """Extends the ``module`` to make it backPACK-ready.

    If the ``module`` has children, e.g. for a ``torch.nn.Sequential``,
    they will also be extended.

    Args:
        module: torch.nn.Module
            The module to extend
        debug: Bool, optional (default: False)
            If true, will print debug messages during the extension.
    """
    if debug:
        print("[DEBUG] Extending", module)

    for child in module.children():
        extend(child, debug=debug)

    module_was_already_extended = getattr(module, "_backpack_extend", False)
    if not module_was_already_extended:
        CTX.add_hook_handle(module.register_forward_hook(hook_store_io))
        CTX.add_hook_handle(module.register_backward_hook(hook_run_extensions))
        module._backpack_extend = True

    return module
