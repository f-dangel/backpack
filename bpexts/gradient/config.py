import torch
from .context import CTX
from .extensions import Extensions
from . import batchgrad, sumgradsquared, diagggn


def set_bpexts(*args):
    """
    Activates the backprop extensions passed as input.
    """
    for arg in args:
        Extensions.check_exists(arg)
    CTX.from_dict({ext: (ext in args) for ext in Extensions.ext_list()})


class bpexts():
    """
    Context manager for the configuration of the backward pass.
    Activates the backprop extensions given as arguments within the context.
    """

    def __init__(self, *args):
        self.args = args

    def __enter__(self):
        self.old_CTX = CTX.as_dict()
        set_bpexts(*self.args)

    def __exit__(self, type, value, traceback):
        CTX.from_dict(self.old_CTX)


def extend(module):
    def store_input(module, input):
        """Pre forward hook saving layer input as buffer.

        Initialize module buffer `ìnput`.
        """
        for i in range(len(input)):
            module.register_buffer('input{}'.format(i),
                                   input[i].clone().detach())

    def store_output(module, input, output):
        """Post-forward hook saving layer output as buffer."""
        module.register_buffer('output', output.clone().detach())

    def store_output_shape(module, input, output):
        """Store dimensionality of output as buffer.

        For debugging and conv/pool operations.
        """
        module.register_buffer('output_shape',
                               torch.IntTensor([*output.size()]))

    def run_extensions(module, grad_input, grad_output):
        """Check which quantities need to be computed and evaluate them."""

        grad_out = [
            grad_output[i].clone().detach() for i in range(len(grad_output))
        ]

        for ext in CTX.active_exts():
            key = (module.__class__, ext)
            if key in Extensions.registeredExtensions:

                print(
                    module.__class__,
                    CTX._backpropagated_sqrt_ggn.shape if hasattr(CTX, "_backpropagated_sqrt_ggn") else None
                )

                Extensions.registeredExtensions[key](module, grad_out)

    CTX.add_hook_handle(module.register_forward_pre_hook(store_input))
    CTX.add_hook_handle(module.register_forward_hook(store_output))
    CTX.add_hook_handle(module.register_forward_hook(store_output_shape))
    CTX.add_hook_handle(module.register_backward_hook(run_extensions))

    return module


def extended(moduleFunc):
    def instanciate_and_extend(*args, **kwargs):
        return extend(moduleFunc(*args, **kwargs))

    return instanciate_and_extend


for ModuleClass, extension, func in batchgrad.SIGNATURE + sumgradsquared.SIGNATURE + diagggn.SIGNATURE:
    Extensions.register(ModuleClass, extension, func)
