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


def test_exts_hooking():
    """Test tracking of hooks registrated with exts functions."""
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


def test_exts_buffering():
    """Test registration of buffers by exts methods."""
    layer = DecoratedAffine(in_features=5, out_features=5)
    # buffer1 is registered normally, buffer2 with exts method
    buffer_names = ['buffer1', 'buffer2']
    tensors = [Tensor([1, 2, 3]), Tensor([4, 5, 6])]
    buffer_register = [layer.register_buffer, layer.register_exts_buffer]
    # register
    for name, tensor, register in zip(buffer_names, tensors, buffer_register):
        register(name, tensor)
    # check if all buffers are present
    for b in buffer_names:
        assert b in layer._buffers
    # but only buffer2 is tracked by exts
    assert buffer_names[0] not in layer.exts_buffers
    assert buffer_names[1] in layer.exts_buffers
