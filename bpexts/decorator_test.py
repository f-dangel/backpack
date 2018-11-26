"""Test decorator of torch.nn.Module subclasses."""

import torch.nn
from .decorator import decorate


def test_decorated_linear_properties():
    """Test name and docstring of decorated torch.nn.Linear."""
    DecoratedAffine = decorate(torch.nn.Linear)
    assert DecoratedAffine.__name__ == 'DecoratedLinear'
    assert DecoratedAffine.__doc__.startswith('[Decorated by bpexts]')


def test_exts_hooking():
    """Test tracking of hooks registrated with exts functions."""
    DecoratedAffine = decorate(torch.nn.Linear)
    layer = DecoratedAffine(in_features=5, out_features=5)

    # dummy hooks
    def dummy_backward_hook(module, grad_input, grad_output):
        """Dummy backward hook."""
        pass

    def dummy_forward_hook(module, input, output):
        """Dummy forward hook."""
        pass

    def dummy_pre_forward_hook(module, input):
        """Dummy pre forward hook."""
        pass

    # exts and nn hook registration functions
    hooks = [dummy_backward_hook,
             dummy_forward_hook,
             dummy_pre_forward_hook]
    exts_register = [layer.register_exts_backward_hook,
                     layer.register_exts_forward_hook,
                     layer.register_exts_forward_pre_hook]
    nn_register = [layer.register_backward_hook,
                   layer.register_forward_hook,
                   layer.register_forward_pre_hook]
    hooks_and_register = zip(hooks, exts_register, nn_register)

    # iterate over hooks and registration methods
    for i, (hook, exts_func, nn_func) in enumerate(hooks_and_register, 1):
        # hooks registered by exts functions should be tracked
        exts_func(hook)
        assert len(layer.exts_hooks) == i
        # usual hook registration should not affect extension hooks
        nn_func(hook)
        assert len(layer.exts_hooks) == i
