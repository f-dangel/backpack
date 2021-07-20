"""BackPACK."""
import inspect
from typing import Callable, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from backpack.extensions.backprop_extension import BackpropExtension
from backpack.utils.hooks import no_op

from . import extensions
from .context import CTX
from .utils import TORCH_VERSION_HIGHER_THAN_1_9_0
from .utils.module_classification import is_no_op


class backpack:
    """Activate BackPACK extensions."""

    def __init__(
        self,
        *exts: BackpropExtension,
        extension_hook: Callable[[Module], None] = None,
        debug: bool = False
    ):
        """Activate BackPACK extensions.

        Enables the BackPACK extensions passed as arguments in the
        :code:`backward` calls inside the current :code:`with` block.

        Args:
            exts: Extensions to activate in the backward pass.
            extension_hook: Function called on each module after
                all BackPACK extensions have run. Takes a ``torch.nn.Module`` and returns
                ``None``. Default: ``None`` (no operation will be formed).
            debug: Print debug messages during the backward pass. Default: ``False``.

        .. note::
            extension_hook can be used to reduce memory overhead if the goal is to compute
            transformations of BackPACK quantities. Information can be compacted
            during a backward pass and obsolete tensors be freed manually (``del``).

        Raises:
            ValueError: if extensions are not valid
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

        self.exts: Tuple[BackpropExtension, ...] = exts
        self.debug: bool = debug
        self.extension_hook: Callable[[Module], None] = (
            no_op if extension_hook is None else extension_hook
        )

    def __enter__(self):
        """Setup backpack environment."""
        self.old_CTX = CTX.get_active_exts()
        self.old_debug = CTX.get_debug()
        self.old_extension_hook = CTX.get_extension_hook()
        CTX.set_active_exts(self.exts)
        CTX.set_debug(self.debug)
        CTX.set_extension_hook(self.extension_hook)

    def __exit__(self, type, value, traceback):
        """Leave backpack environment.

        Args:
            type: .
            value: .
            traceback: .
        """
        for backprop_extension in CTX.get_active_exts():
            backprop_extension.clear()
        CTX.set_active_exts(self.old_CTX)
        CTX.set_debug(self.old_debug)
        CTX.set_extension_hook(self.old_extension_hook)


class disable:
    """Entirely disable BackPACK, including storage of input and output.

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

    store_io: bool = True

    def __enter__(self):
        """Disable input/output storing."""
        self.old_store_io: bool = disable.store_io
        disable.store_io = False

    def __exit__(self, type, value, traceback):
        """Set input/output storing to old value.

        Args:
            type: .
            value: .
            traceback: .
        """
        disable.store_io = self.old_store_io

    @staticmethod
    def should_store_io() -> bool:
        """Return whether input and output should be stored.

        Returns:
            whether input and output should be stored
        """
        return disable.store_io


def hook_store_io(
    module: Module, input: Tuple[Tensor], output: Union[Tensor, Tuple[Tensor]]
) -> None:
    """Saves the input and output as attributes of the module.

    The list of inputs with index i is saved as module.input[i]
    The output is reduced to single output tensor and saved as module.output

    Args:
        module: the module on which to save the params
        input: List of input tensors
        output: result of module(input)
    """
    if disable.should_store_io() and torch.is_grad_enabled():
        for i in range(len(input)):
            setattr(module, "input{}".format(i), input[i])
        if isinstance(output, tuple):
            # is true for RNN,GRU,LSTM which return tuple (output, ...)
            module.output = output[0]
        else:
            module.output = output


def memory_cleanup(module) -> None:
    """Remove I/O stored by backpack during the forward pass.

    Deletes the attributes created by `hook_store_io`.

    Args:
        module: current module
    """
    if hasattr(module, "output"):
        delattr(module, "output")
    i = 0
    while hasattr(module, "input{}".format(i)):
        delattr(module, "input{}".format(i))
        i += 1


def hook_run_extensions(
    module: Module, g_inp: Tuple[Tensor], g_out: Tuple[Tensor]
) -> None:
    """The backward hook function.

    It executes all BackPACK operations during the backward pass.

    Args:
        module: current module
        g_inp: input gradients
        g_out: output gradients
    """
    for backpack_extension in CTX.get_active_exts():
        if CTX.get_debug():
            print("[DEBUG] Running extension", backpack_extension, "on", module)
        backpack_extension.apply(module, g_inp, g_out)

    post_extension_hook(module)

    if not (
        CTX.is_extension_active(
            extensions.curvmatprod.HMP,
            extensions.curvmatprod.GGNMP,
            extensions.curvmatprod.PCHMP,
        )
    ):
        memory_cleanup(module)


def post_extension_hook(module: Module) -> None:
    """Execute the post extensions hook on a module after all BackPACK extensions.

    See the `post_backward_hook` argument of the `backpack` context manager for details.

    Args:
        module: current module

    Raises:
        RuntimeError: if any error occurred during the backward hook
    """
    try:
        CTX.get_extension_hook()(module)
    except Exception as e:
        message = getattr(e, "message", repr(e))
        raise RuntimeError(f"Post extensions hook failed: {message}")


def extend(module: Module, debug: bool = False) -> Module:
    """Extends a ``module`` to make it BackPACK-ready.

    If the ``module`` has children, e.g. for a ``torch.nn.Sequential``,
    they will also be extended.

    Args:
        module: The module to extend.
        debug: Print debug messages during the extension. Default: ``False``.

    Returns:
        Extended module.
    """
    if debug:
        print("[DEBUG] Extending", module)

    for child in module.children():
        extend(child, debug=debug)

    module_was_already_extended = getattr(module, "_backpack_extend", False)
    if not module_was_already_extended:
        CTX.add_hook_handle(module.register_forward_hook(hook_store_io))
        _register_hooks(module)
        module._backpack_extend = True

    return module


def _register_hooks(module: Module) -> None:
    if is_no_op(module):
        return

    if TORCH_VERSION_HIGHER_THAN_1_9_0:
        CTX.add_hook_handle(module.register_full_backward_hook(hook_run_extensions))
    else:
        CTX.add_hook_handle(module.register_backward_hook(hook_run_extensions))
