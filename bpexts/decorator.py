"""Decorator for instances of torch.nn.Module subclasses.

Adds methods to a usual layer used in PyTorch without reimplementing
the backward/forward pass.
"""

import torch.nn


def decorate(module_subclass):
    """Extend functionality of a module.

    The functionality allows to install

    - hooks whose book-keeping decoupled from the normal hooks and
      allows them to be removed if desired.

    - buffers whose book-keeping is decoupled from the normal buffers

    Implemented in this way, each subclass of :obj:`torch.nn.Module`
    is decorated separately, thereby avoiding double-inheritance from
    :obj:`torch.nn.Module` in a diamond-like pattern.

    Parameters
    ----------
    module_subclass : torch.nn.Module subclass
        A subclass of :obj:`torch.nn.Module `that will be extended
        by the aforementioned functionality

    Returns
    -------
    class
        Can be used exactly the same as the original class
        but has additional methods to implement extensions of
        backpropagation
    """
    if not issubclass(module_subclass, torch.nn.Module):
        raise ValueError('Can onĺy wrap subclasses of torch.nn.Module')

    class DecoratedModule(module_subclass):
        """Module decorated for backpropagation extension.

        Attributes
        ----------
        exts_hooks : :obj:`list`
            Stores handles of hooks that allow for easy removal
        exts_buffers : :obj:`set`
            Store buffers of backprop extension
        """
        __doc__ = '[Decorated by bpexts] {}'.format(module_subclass.__doc__)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.exts_hooks = []
            self.exts_buffers = set()

        __init__.__doc__ = module_subclass.__init__.__doc__

        # --- hook tracking ---
        def register_exts_forward_pre_hook(self, hook):
            """Register forward pre hook for extended backpropgation."""
            handle = self.register_forward_pre_hook(hook)
            return self._track_exts_handle(handle)

        def register_exts_forward_hook(self, hook):
            """Register forward hook for extended backpropgation."""
            handle = self.register_forward_hook(hook)
            return self._track_exts_handle(handle)

        def register_exts_backward_hook(self, hook):
            """Register backward hook for extended backpropgation."""
            handle = self.register_backward_hook(hook)
            return self._track_exts_handle(handle)

        def _track_exts_handle(self, handle):
            """Track hooks set up for extended backpropagation."""
            self.exts_hooks.append(handle)
            return handle

        # --- buffer tracking ---
        def register_exts_buffer(self, name, tensor=None):
            """Register tracked buffer for extended backpropagation.

            If no tensor is specified, `None` is used as a placeholder.

            Parameters
            ----------
            name : :obj:`str`
                Name of the buffer
            tensor : torch.Tensor, optional
                Tensor to store as buffer
            """
            self.register_buffer(name, tensor)
            self._track_exts_buffer(name)

        def _track_exts_buffer(self, name):
            """Keep track of buffers."""
            self.exts_buffers.add(name)

        # --- disable exts ---
        def disable_exts(self, keep_buffers=False):
            """Disable exts behavior, make module behave like torch.nn.

            Parameters
            ----------
            keep_buffers: :obj:`bool`
                Also remove buffers stored by backpropagation extension
            """
            self.remove_exts_hooks()
            if not keep_buffers:
                self.remove_exts_buffers()
            for module in self.children():
                module.disable_exts(keep_buffers=keep_buffers)

        def remove_exts_hooks(self):
            """Remove all hooks tracked by bpexts."""
            while self.exts_hooks:
                handle = self.exts_hooks.pop()
                handle.remove()

        def remove_exts_buffers(self):
            """Remove all buffers introduced by bpexts."""
            while self.exts_buffers:
                name = self.exts_buffers.pop()
                self._buffers.pop(name)

        def extra_repr(self):
            """Show number of active extension buffers and hooks."""
            active_hooks = len(self.exts_hooks)
            active_buffers = len(self.exts_buffers)
            repr = '{}{}buffers: {}, hooks {}'.format(
                super().extra_repr(), ', ' if super().extra_repr() else '',
                active_buffers, active_hooks)
            return repr

    DecoratedModule.__name__ = 'Decorated{}'.format(module_subclass.__name__)

    return DecoratedModule
