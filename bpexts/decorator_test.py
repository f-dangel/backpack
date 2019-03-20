"""Test decorator of torch.nn.Module subclasses."""

from torch import Tensor
from torch.nn import Linear
from .decorator import decorate


# decorated linear layer
DecoratedAffine = decorate(Linear)


def test_decorated_linear_properties():
    """Test name and docstring of decorated torch.nn.Linear."""
    assert DecoratedAffine.__name__ == 'DecoratedLinear'
    assert DecoratedAffine.__doc__.startswith('[Decorated by bpexts]')


def dummy_backward_hook(module, grad_input, grad_output):
    """Dummy backward hook."""
    pass


def dummy_forward_hook(module, input, output):
    """Dummy forward hook."""
    pass


def dummy_pre_forward_hook(module, input):
    """Dummy pre forward hook."""
    pass


def test_exts_hooking():
    """Test tracking of hooks registrated with exts functions.

    Return module with hooks (6 in total, 3 registered by torch.nn,
    3 registered by exts.
    """
    layer = DecoratedAffine(in_features=5, out_features=5)
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
    # each hook type occurs twice (once from nn, once from exts)
    for hook_type in [layer._backward_hooks,
                      layer._forward_hooks,
                      layer._forward_pre_hooks]:
        assert len(hook_type) == 2
    return layer


def module_with_hooks():
    """Return module with 6 hooks.

    Module contains a backward, a forward and a pre forward hook.
    The hooks are registered using the torch.nn method and the
    exts method, resultin in 2*3 = 6 hooks.
    """
    return test_exts_hooking()


def test_hook_removing():
    """Test disabling of hooks tracked by exts."""
    # 6 hooks, 3 tracked by exts, 3 tracked by torch.nn
    layer = module_with_hooks()
    # remove tracked hooks
    layer.remove_exts_hooks()
    assert len(layer.exts_hooks) == 0
    # module has one forward, pre forward and backward hook each.
    for hook_type in [layer._backward_hooks,
                      layer._forward_hooks,
                      layer._forward_pre_hooks]:
        assert len(hook_type) == 1


def module_with_buffers():
    """Return module that holds two buffers.

    `'buffer1'` is registered by the torch.nn.Module method.
    `'buffer2'` is registered by the exts method.
    """
    layer = DecoratedAffine(in_features=5, out_features=5)
    # buffer1 is registered normally, buffer2 with exts method
    buffer_names = ['buffer1', 'buffer2']
    tensors = [Tensor([1, 2, 3]), Tensor([4, 5, 6])]
    buffer_register = [layer.register_buffer, layer.register_exts_buffer]
    # register
    for name, tensor, register in zip(buffer_names, tensors, buffer_register):
        register(name, tensor)
    return layer, buffer_names


def test_exts_buffering():
    """Test registration of buffers by exts methods."""
    layer, buffer_names = module_with_buffers()
    # check if all buffers are present
    for b in buffer_names:
        assert b in layer._buffers
    # but only buffer2 is tracked by exts
    assert buffer_names[0] not in layer.exts_buffers
    assert buffer_names[1] in layer.exts_buffers


def test_buffer_removing():
    """Test removing of exts buffers."""
    layer, buffer_names = module_with_buffers()
    # remove buffer_names[1] which was registered with exts
    layer.remove_exts_buffers()
    # buffer1 is still existent
    assert buffer_names[0] in layer._buffers
    assert hasattr(layer, buffer_names[0])
    # buffer2 not
    assert not (buffer_names[1] in layer._buffers)
    assert not hasattr(layer, buffer_names[1])
