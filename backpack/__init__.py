"""Computation of batch gradients.

Computing parameter gradients for each batch sample can be used
to calculate the variance of a gradient.
"""
import torch
from .secondorder import diagh, hbp
from .extensions import Extension, Extensions
from . import curvmatprod as cmp
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
        self.old_CTX = CTX.active_exts() + CTX.new_active_exts()
        CTX.set_active_exts(self.args)

    def __exit__(self, type, value, traceback):
        CTX.set_active_exts(self.old_CTX)
        CTX.clear()


def has_children(mod):
    return len(list(mod.children())) > 0


def extend(module):
    if DEBUGGING:
        print("[DEBUG] Extending", module)

    if has_children(module):
        for child in module.children():
            extend(child)

    def store_io(module, input, output):
        for i in range(len(input)):
            setattr(module, 'input{}'.format(i), input[i])
        setattr(module, 'output', output)

    def store_shapes(module, input, output):
        """Store dimensionality of output as buffer."""
        for i in range(len(input)):
            module.register_buffer('input{}_shape'.format(i),
                                   torch.IntTensor([*input[i].size()]))
        module.register_buffer('output_shape',
                               torch.IntTensor([*output.size()]))

    def run_extensions(module, grad_input, grad_output):
        """Check which quantities need to be computed and evaluate them."""
        if DEBUGGING:
            print("[DEBUG] Backward Hook called on [{}]".format(module))
            if len(CTX.active_exts()) == 0:
                print("[DEBUG] No Active Extension")
            else:
                print("[DEBUG] Extensions active: {}".format(
                    CTX.active_exts()))

        grad_out = [grad_output[i] for i in range(len(grad_output))]

        for backpack_extension in CTX.new_active_exts():
            print(" └───────[DEBUG] New Backward hook {}-{}".format(module.__class__, backpack_extension.__class__))
            backpack_extension.apply(module, grad_input, grad_output)

        for extension in CTX.active_exts():

            exts_for_mod = list(
                Extensions.get_extensions_for([extension], module))

            if DEBUGGING and len(exts_for_mod) == 0:
                print(" └─[DEBUG] No extension registered for {}".format(
                    module.__class__))

            for bpext in exts_for_mod:
                if DEBUGGING:
                    print(" └─[DEBUG] Backward hook {}".format(
                        bpext.__class__, ))

                bpext.apply(extension, module, grad_input, grad_out)

    CTX.add_hook_handle(module.register_forward_hook(store_io))
    CTX.add_hook_handle(module.register_forward_hook(store_shapes))
    CTX.add_hook_handle(module.register_backward_hook(run_extensions))

    return module


def extended(moduleFunc):
    def instanciate_and_extend(*args, **kwargs):
        return extend(moduleFunc(*args, **kwargs))

    return instanciate_and_extend


EXTENSIONS = [
    *diagh.EXTENSIONS,
    *cmp.EXTENSIONS,
    *hbp.EXTENSIONS,
]

for backpropextension in EXTENSIONS:
    Extensions.register(backpropextension)
