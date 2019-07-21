"""Computation of batch gradients.

Computing parameter gradients for each batch sample can be used
to calculate the variance of a gradient.
"""
import torch
from .context import CTX

DEBUGGING = True


class backpack():
    """
    Context manager for the configuration of the backward pass.
    Activates the backprop extensions given as arguments within the context.
    """

    def __init__(self, *args):
        self.args = args

    def __enter__(self):
        self.old_CTX = CTX.get_active_exts()
        CTX.set_active_exts(self.args)

    def __exit__(self, type, value, traceback):
        CTX.set_active_exts(self.old_CTX)
        CTX.clear()


def extend(module):
    if DEBUGGING:
        print("[DEBUG] Extending", module)

    for child in module.children():
        extend(child)

    def store_io(module, input, output):
        for i in range(len(input)):
            setattr(module, 'input{}'.format(i), input[i])
        setattr(module, 'output', output)

    def store_shapes(module, input, output):
        """Store dimensionality of output as buffer."""
        for i in range(len(input)):
            module.register_buffer(
                'input{}_shape'.format(i),
                torch.IntTensor([*input[i].size()])
            )
        module.register_buffer(
            'output_shape',
            torch.IntTensor([*output.size()])
        )

    def run_extensions(module_, g_inp, g_out):
        for backpack_extension in CTX.get_active_exts():
            if DEBUGGING:
                print(
                    "[DEBUG] Running extension", backpack_extension,
                    "on ", module
                )
            backpack_extension.apply(module_, g_inp, g_out)

    CTX.add_hook_handle(module.register_forward_hook(store_io))
    CTX.add_hook_handle(module.register_forward_hook(store_shapes))
    CTX.add_hook_handle(module.register_backward_hook(run_extensions))

    return module
