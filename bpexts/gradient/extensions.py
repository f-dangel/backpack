import warnings
from .backpropextension import BackpropExtension


class Extension():
    def __init__(self):
        return


class VARIANCE(Extension):
    pass


class BATCH_L2(Extension):
    pass


class BATCH_GRAD(Extension):
    pass


class SUM_GRAD_SQUARED(Extension):
    pass


class GRAD_VAR(Extension):
    pass


class DIAG_GGN(Extension):
    pass


class GRAD(Extension):
    pass


class ParametrizedExtension(Extension):
    def __init__(self, input):
        self.input = input


class JVP(Extension):
    pass


class Extensions:

    EXTENSIONS = [
        GRAD,
        BATCH_GRAD,
        SUM_GRAD_SQUARED,
        GRAD_VAR,
        DIAG_GGN,
        BATCH_L2,
        VARIANCE,
        JVP,
    ]

    registeredExtensions = {}

    @staticmethod
    def ext_list():
        return Extensions.EXTENSIONS

    @staticmethod
    def register(backpropextension):
        assert isinstance(backpropextension, BackpropExtension)

        Extensions.check_exists(backpropextension._BackpropExtension__get_ext())

        key = backpropextension._BackpropExtension__get_key()

        already_exist = key in Extensions.registeredExtensions
        if already_exist:
            warnings.warn("Extension {} for layer {} already registered".format(key[1], key[0]), category=RuntimeWarning)

        Extensions.registeredExtensions[key] = backpropextension

    @staticmethod
    def check_exists(ext):
        if ext not in Extensions.EXTENSIONS:
            raise ValueError("Backprop extension [{}] unknown".format(ext))

    @staticmethod
    def get_extensions_for(active_exts, module):
        for ext in active_exts:
            key = (module.__class__, ext)
            if key in Extensions.registeredExtensions:
                yield Extensions.registeredExtensions[key]
