"""Computation of batch gradients.

Computing parameter gradients for each batch sample can be used
to calculate the variance of a gradient.
"""
import torch
from .firstorder import batchgrad, sumgradsquared, batchl2, variance
from .secondorder import diagggn, diagh, kflr, kfac, cmp
from .extensions import Extension, Extensions
from .context import CTX

DEBUGGING = True


def set_backpack(*args):
    """
    Activates the backprop extensions passed as input.
    """
    for arg in args:
        Extensions.check_exists(arg)

    args_classes = []
    for arg in args:
        if isinstance(arg, Extension):
            args_classes.append(arg.__class__)
        else:
            args_classes.append(arg)

    CTX.from_dict(
        {ext: (ext in args_classes)
         for ext in Extensions.ext_list()})


class backpack():
    """
    Context manager for the configuration of the backward pass.
    Activates the backprop extensions given as arguments within the context.
    """

    def __init__(self, *args):
        self.args = args

    def __enter__(self):
        self.old_CTX = CTX.as_dict()
        set_backpack(*self.args)

    def __exit__(self, type, value, traceback):
        CTX.from_dict(self.old_CTX)


def extend(module):

    if DEBUGGING:
        print("[DEBUG] Extending", module)

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
            print("[DEBUG] Extensions active: {}".format(CTX.active_exts()))

        grad_out = [grad_output[i] for i in range(len(grad_output))]

        exts_for_mod = list(
            Extensions.get_extensions_for(CTX.active_exts(), module))

        if DEBUGGING and len(exts_for_mod) == 0:
            print(" └─[DEBUG] No extension registered for {}".format(
                module.__class__))

        for bpext in exts_for_mod:
            if DEBUGGING:
                print(" └─[DEBUG] Backward hook {}".format(bpext.__class__, ))

            bpext.apply(module, grad_input, grad_out)

    CTX.add_hook_handle(module.register_forward_hook(store_io))
    CTX.add_hook_handle(module.register_forward_hook(store_shapes))
    CTX.add_hook_handle(module.register_backward_hook(run_extensions))

    return module


def extended(moduleFunc):
    def instanciate_and_extend(*args, **kwargs):
        return extend(moduleFunc(*args, **kwargs))

    return instanciate_and_extend


EXTENSIONS = [
    *batchgrad.EXTENSIONS,
    *sumgradsquared.EXTENSIONS,
    *diagggn.EXTENSIONS,
    *batchl2.EXTENSIONS,
    *variance.EXTENSIONS,
    *diagh.EXTENSIONS,
    *kflr.EXTENSIONS,
    *kfac.EXTENSIONS,
    *cmp.EXTENSIONS,
]

for backpropextension in EXTENSIONS:
    Extensions.register(backpropextension)
