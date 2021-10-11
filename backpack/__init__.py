"""BackPACK."""
from inspect import isclass
from types import TracebackType
from typing import Callable, Optional, Tuple, Type, Union

from torch import Tensor, is_grad_enabled
from torch.fx import GraphModule
from torch.nn import Module

from backpack import extensions
from backpack.context import CTX
from backpack.custom_module.graph_utils import convert_module_to_backpack
from backpack.extensions.backprop_extension import BackpropExtension
from backpack.utils.hooks import no_op
from backpack.utils.module_classification import is_no_op


class backpack:
    """Context manager to activate BackPACK extensions."""

    def __init__(
        self,
        *exts: BackpropExtension,
        extension_hook: Callable[[Module], None] = None,
        debug: bool = False,
        retain_graph: bool = False,
    ):
        """Activate BackPACK extensions.

        Enables the BackPACK extensions passed as arguments in the
        :code:`backward` calls inside the current :code:`with` block.

        Args:
            exts: Extensions to activate in the backward pass.
            extension_hook: Function called on each module after
                all BackPACK extensions have run. Takes a ``torch.nn.Module`` and returns
                ``None``. Default: ``None`` (no operation will be performed).
            debug: Print debug messages during the backward pass. Default: ``False``.
            retain_graph: Determines whether BackPack IO should be kept for additional
                backward passes. Should have same value as the argument ``retain_graph``
                in ``backward()``. Default: ``False``.

        .. note::
            extension_hook can be used to reduce memory overhead if the goal is to compute
            transformations of BackPACK quantities. Information can be compacted
            during a backward pass and obsolete tensors be freed manually (``del``).

        Raises:
            ValueError: if extensions are not valid
        """
        for ext in exts:
            if not isinstance(ext, BackpropExtension):
                if isclass(ext) and issubclass(ext, BackpropExtension):
                    raise ValueError(
                        "backpack expects instances of BackpropExtension,"
                        + f" but received a class instead [{ext}]."
                        + " Instantiate it before passing it to backpack."
                    )
                else:
                    raise ValueError(
                        "backpack expects instances of BackpropExtension,"
                        + f" but received [{ext}]."
                    )

        self.exts: Tuple[BackpropExtension, ...] = exts
        self.debug: bool = debug
        self.extension_hook: Callable[[Module], None] = (
            no_op if extension_hook is None else extension_hook
        )
        self.retain_graph = retain_graph

    def __enter__(self):
        """Setup backpack environment."""
        self.old_CTX = CTX.get_active_exts()
        self.old_debug = CTX.get_debug()
        self.old_extension_hook = CTX.get_extension_hook()
        self.old_retain_graph = CTX.get_retain_graph()
        CTX.set_active_exts(self.exts)
        CTX.set_debug(self.debug)
        CTX.set_extension_hook(self.extension_hook)
        CTX.set_retain_graph(self.retain_graph)

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ):
        """Leave backpack environment.

        Args:
            __exc_type: exception type
            __exc_value: exception value
            __traceback: exception traceback
        """
        CTX.set_active_exts(self.old_CTX)
        CTX.set_debug(self.old_debug)
        CTX.set_extension_hook(self.old_extension_hook)
        CTX.set_retain_graph(self.old_retain_graph)


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

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ):
        """Leave backpack environment.

        Args:
            __exc_type: exception type
            __exc_value: exception value
            __traceback: exception traceback
        """
        disable.store_io = self.old_store_io

    @staticmethod
    def should_store_io() -> bool:
        """Return whether input and output should be stored during forward pass.

        Returns:
            whether input and output should be stored during forward pass
        """
        return disable.store_io


def hook_store_io(
    module: Module, input: Tuple[Tensor], output: Union[Tensor, Tuple[Tensor]]
) -> None:
    """Saves the input and output as attributes of the module.

    The list of inputs with index i is saved as module.input[i]
    The output is reduced to single output tensor and saved as module.output

    Args:
        module: the module on which to save the inputs/outputs
        input: List of input tensors
        output: result of module(input)
    """
    if disable.should_store_io() and is_grad_enabled():
        for i in range(len(input)):
            setattr(module, "input{}".format(i), input[i])
        if isinstance(output, tuple):
            # is true for RNN,GRU,LSTM which return tuple (output, ...)
            module.output = output[0]
        else:
            module.output = output


def memory_cleanup(module: Module) -> None:
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
    debug = CTX.get_debug()
    for backpack_extension in CTX.get_active_exts():
        if debug:
            print("[DEBUG] Running extension", backpack_extension, "on", module)
        backpack_extension(module, g_inp, g_out)

    if debug:
        print("[DEBUG] Running extension hook on", module)
    CTX.get_extension_hook()(module)

    if not (
        CTX.get_retain_graph()
        or (
            CTX.is_extension_active(
                extensions.curvmatprod.HMP,
                extensions.curvmatprod.GGNMP,
                extensions.curvmatprod.PCHMP,
            )
        )
    ):
        memory_cleanup(module)


def extend(module: Module, debug: bool = False, use_converter: bool = False) -> Module:
    """Recursively extend a ``module`` to make it BackPACK-ready.

    Modules that do not represent an operation in the computation graph (for instance
    containers like ``Sequential``) will not explicitly be extended.

    Args:
        module: The module to extend.
        debug: Print debug messages during the extension. Default: ``False``.
        use_converter: Try converting the module to a BackPACK-compatible network.
            The converter might alter the model, e.g. order of parameters.
            Default: ``False``.

    Returns:
        Extended module.
    """
    if debug:
        print("[DEBUG] Extending", module)

    if use_converter:
        module: GraphModule = convert_module_to_backpack(module, debug)
        return extend(module)

    for child in module.children():
        extend(child, debug=debug)

    extended_flag = "_backpack_extend"
    already_extended = getattr(module, extended_flag, False)
    if not (already_extended or is_no_op(module)):
        _register_hooks(module)
        setattr(module, extended_flag, True)

    return module


def _register_hooks(module: Module) -> None:
    """Install forward and backward hooks on a module.

    Args:
          module: module that is going to be extended
    """
    CTX.add_hook_handle(module.register_forward_hook(hook_store_io))
    CTX.add_hook_handle(module.register_full_backward_hook(hook_run_extensions))
