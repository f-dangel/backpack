"""Decorator for instances of torch.nn.Module subclasses.

Allows to add methods to a usual layer used in torch without
having to reimplement backward/forward.
"""

from torch.nn import Module


def decorate(module_subclass):
    """Add functionality to torch.nn.Module subclass.

    Implemented in this way, each subclass of nn.Module is decorated
    separately, thereby avoiding double-inheritance from torch.nn.Module
    in a diamond-like pattern.
    """
    if not issubclass(module_subclass, Module):
        raise ValueError('Can onĺy wrap subclasses of torch.nn.Module')

    class DecoratedModule(module_subclass):
        """Module decorated for backpropagation extension."""
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
            """Register tracked buffer for extended backpropagation."""
            self.register_buffer(name, tensor)
            self._track_exts_buffer(name)

        def _track_exts_buffer(self, name):
            """Keep track of buffers."""
            self.exts_buffers.add(name)

        # --- disable exts ---
        def disable_exts(self):
            """Disable exts behavior, make module behave like torch.nn."""
            self.remove_exts_buffers()
            self.remove_exts_hooks()

        def remove_exts_hooks(self):
            """Remove all hooks tracked by exts."""
            while self.exts_hooks:
                handle = self.exts_hooks.pop()
                handle.remove()

        def remove_exts_buffers(self):
            """Remove all buffers introduced by exts."""
            while self.exts_buffers:
                name = self.exts_buffers.pop()
                self._buffers.pop(name)

    DecoratedModule.__name__ = 'Decorated{}'.format(module_subclass.__name__)

    return DecoratedModule
