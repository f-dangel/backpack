"""Context class for BackPACK."""
from typing import Callable, Iterable, List, Tuple, Type

from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from backpack.extensions.backprop_extension import BackpropExtension
from backpack.utils.hooks import no_op


class CTX:
    """Global Class holding the configuration of the backward pass."""

    active_exts: Tuple[BackpropExtension] = tuple()
    debug: bool = False
    extension_hook: Callable[[Module], None] = no_op
    hook_handles: List[RemovableHandle] = []
    retain_graph: bool = False

    @staticmethod
    def set_active_exts(active_exts: Iterable[BackpropExtension]) -> None:
        """Set the active backpack extensions.

        Args:
            active_exts: the extensions
        """
        CTX.active_exts = tuple(active_exts)

    @staticmethod
    def get_active_exts() -> Tuple[BackpropExtension]:
        """Get the currently active extensions.

        Returns:
            active extensions
        """
        return CTX.active_exts

    @staticmethod
    def add_hook_handle(hook_handle: RemovableHandle) -> None:
        """Add the hook handle to internal variable hook_handles.

        Args:
            hook_handle: the removable handle
        """
        CTX.hook_handles.append(hook_handle)

    @staticmethod
    def remove_hooks() -> None:
        """Remove all hooks."""
        for handle in CTX.hook_handles:
            handle.remove()
        CTX.hook_handles = []

    @staticmethod
    def is_extension_active(*extension_classes: Type[BackpropExtension]) -> bool:
        """Returns whether the specified class is currently active.

        Args:
            *extension_classes: classes to test

        Returns:
            whether at least one of the specified extensions is active
        """
        return any(isinstance(ext, extension_classes) for ext in CTX.get_active_exts())

    @staticmethod
    def get_debug() -> bool:
        """Whether debug mode is active.

        Returns:
            whether debug mode is active
        """
        return CTX.debug

    @staticmethod
    def set_debug(debug: bool) -> None:
        """Set debug mode.

        Args:
            debug: the mode to set
        """
        CTX.debug = debug

    @staticmethod
    def get_extension_hook() -> Callable[[Module], None]:
        """Return the current extension hook to be run after all other extensions.

        Returns:
            current extension hook
        """
        return CTX.extension_hook

    @staticmethod
    def set_extension_hook(extension_hook: Callable[[Module], None]) -> None:
        """Set the current extension hook.

        Args:
            extension_hook: the extension hook to run after all other extensions
        """
        CTX.extension_hook = extension_hook

    @staticmethod
    def set_retain_graph(retain_graph: bool) -> None:
        """Set retain_graph.

        Args:
            retain_graph: new value for retain_graph
        """
        CTX.retain_graph = retain_graph

    @staticmethod
    def get_retain_graph() -> bool:
        """Get retain_graph.

        Returns:
            retain_graph
        """
        return CTX.retain_graph
