import warnings
from .backpropextension import BackpropExtension
from .curvature import Curvature


class Extension():
    def __init__(self):
        return


class VARIANCE(Extension):
    savefield = "variance"
    pass


class BATCH_L2(Extension):
    savefield = "batch_l2"
    pass


class BATCH_GRAD(Extension):
    savefield = "grad_batch"
    pass


class SUM_GRAD_SQUARED(Extension):
    savefield = "sum_grad_squared"
    pass


class DIAG_GGN(Extension):
    savefield = "diag_ggn"
    pass


class GRAD(Extension):
    savefield = "grad_bpext"
    pass


class DIAG_H(Extension):
    savefield = "diag_h"
    pass


class KFLR(Extension):
    savefield = "kflr"
    pass


class KFAC(Extension):
    savefield = "kfac"
    pass


class KFRA(Extension):
    savefield = "kfra"
    pass


class ParametrizedExtension(Extension):
    def __init__(self, input):
        self.input = input


class CMP(ParametrizedExtension):
    savefield = "cmp"

    def __init__(self, which):
        Curvature.set_current(which)
        super().__init__(which)


class JVP(Extension):
    pass


class Extensions:

    EXTENSIONS = [
        GRAD,
        BATCH_GRAD,
        SUM_GRAD_SQUARED,
        DIAG_GGN,
        BATCH_L2,
        VARIANCE,
        JVP,
        DIAG_H,
        KFLR,
        KFRA,
        KFAC,
        CMP,
    ]

    registeredExtensions = {}

    @staticmethod
    def ext_list():
        return Extensions.EXTENSIONS

    @staticmethod
    def register(backpropextension: BackpropExtension):

        Extensions.check_exists(
            backpropextension._BackpropExtension__get_ext())

        key = backpropextension._BackpropExtension__get_key()

        already_exist = key in Extensions.registeredExtensions
        if already_exist:
            warnings.warn(
                "Extension {} for layer {} already registered".format(
                    key[1], key[0]),
                category=RuntimeWarning)

        Extensions.registeredExtensions[key] = backpropextension

    @staticmethod
    def check_exists(ext):
        ext_cls = ext.__class__ if isinstance(ext, Extension) else ext
        if ext_cls not in Extensions.EXTENSIONS:
            raise ValueError("Backprop extension [{}] unknown".format(ext_cls))

    @staticmethod
    def get_extensions_for(active_exts, module):
        for ext in active_exts:
            key = (module.__class__, ext)
            if key in Extensions.registeredExtensions:
                yield Extensions.registeredExtensions[key]
