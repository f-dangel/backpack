"""Decorator for instances of torch.nn.Module subclasses.

Allows to add methods to a usual layer used in torch without
having to reimplement backward/forward.
"""

from torch.nn import Module


def decorate(module_subclass):
    """Add functionality to torch.nn.Module subclass."""
    if not issubclass(module_subclass, Module):
        raise ValueError('Can onĺy wrap subclasses of torch.nn.Module')

    class DecoratedModule(module_subclass):
        """Module decorated for backpropagation extension."""
        __doc__ = '[Decorated by bpexts] {}'.format(module_subclass.__doc__)

    DecoratedModule.__name__ = 'Decorated{}'.format(module_subclass.__name__)

    return DecoratedModule
