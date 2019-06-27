import warnings
from . import batchgrad


GRAD = "GRAD"
BATCH_GRAD = "BATCH_GRAD"
SUM_GRAD_SQUARED = "SUM_GRAD_SQUARED"
GRAD_VAR = "GRAD_VAR"
DIAG_GGN = 'DIAG_GGN'


class Extensions:

    EXTENSIONS = [
        GRAD,
        BATCH_GRAD,
        SUM_GRAD_SQUARED,
        GRAD_VAR,
        DIAG_GGN,
    ]

    registeredExtensions = {}

    @staticmethod
    def ext_list():
        return Extensions.EXTENSIONS

    @staticmethod
    def register(LayerClass, ext, func):
        Extensions.check_exists(ext)
        key = (LayerClass, ext)
        print("REGISTER")
        if key in Extensions.registeredExtensions:
            warnings.warn("Extension {} for layer {} already registered".format(ext, LayerClass), category=RuntimeWarning)
        Extensions.registeredExtensions[key] = func

    @staticmethod
    def check_exists(ext):
        if ext not in Extensions.EXTENSIONS:
            raise ValueError("Backprop extension [{}] unknown".format(ext))


class CTX:
    """
    Global Class holding the configuration of the backward pass
    """

    @staticmethod
    def as_dict():
        return {ext: getattr(CTX, ext, False) for ext in Extensions.ext_list()}

    @staticmethod
    def from_dict(dic):
        for key, val in dic.items():
            setattr(CTX, key, val)

    @staticmethod
    def is_active(ext):
        Extensions.check_exists(ext)
        return getattr(CTX, ext, False)

    @staticmethod
    def active_exts():
        return [ext for ext, active in CTX.as_dict().items() if active]


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
            module.register_buffer('input{}'.format(i), input[i].clone().detach())
            # store output
            # module.register_buffer('input{}'.format(i), input[i].clone().detach())

    def run_extensions(module, grad_input, grad_output):
        """Check which quantities need to be computed and evaluate them."""

        grad_out = [grad_output[i].clone().detach() for i in range(len(grad_output))]

        for ext in CTX.active_exts():
            key = (module.__class__, ext)
            if key in Extensions.registeredExtensions:
                Extensions.registeredExtensions[key](module, grad_out)

    module.register_forward_pre_hook(store_input)
    module.register_backward_hook(run_extensions)

    return module


for ModuleClass, extension, func in batchgrad.SIGNATURE:
    Extensions.register(ModuleClass, extension, func)
