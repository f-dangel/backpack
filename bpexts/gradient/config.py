GRAD = "grad"
BATCH_GRAD = "batch_grad"
SUM_GRAD_SQUARED = "SGS"
GRAD_VAR = "grad_var"

EXTENSIONS = [
    GRAD,
    BATCH_GRAD,
    SUM_GRAD_SQUARED,
    GRAD_VAR
]


class CTX:
    """
    Global Class holding the configuration of the backward pass
    """
    @staticmethod
    def as_dict():
        return {ext: getattr(CTX, ext, False) for ext in EXTENSIONS}

    @staticmethod
    def from_dict(dic):
        for key, val in dic.items():
            setattr(CTX, key, val)


def set_bpexts(*args):
    """
    Activates the backprop extensions passed as input.
    """
    for arg in args:
        assert arg in EXTENSIONS
    CTX.from_dict({ext: (ext in args) for ext in EXTENSIONS})


class bpexts():
    """
    Context manager for the configuration of the backward pass.
    Activates the backprop extensions given as arguments within the context.
    """

    def __init__(self, *args):
        self.args = args

    def __enter__(self):
        self.old_CTX = CTX.as_dict()
        set_bpexts(self.args)

    def __exit__(self, type, value, traceback):
        CTX.from_dict(self.old_CTX)
