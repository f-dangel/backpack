import warnings

GRAD = "GRAD"
BATCH_GRAD = "BATCH_GRAD"
SUM_GRAD_SQUARED = "SUM_GRAD_SQUARED"
GRAD_VAR = "GRAD_VAR"
DIAG_GGN = 'DIAG_GGN'


class Extensions:

    GRAD = "GRAD"
    BATCH_GRAD = "BATCH_GRAD"
    SUM_GRAD_SQUARED = "SUM_GRAD_SQUARED"
    GRAD_VAR = "GRAD_VAR"
    DIAG_GGN = 'DIAG_GGN'

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
