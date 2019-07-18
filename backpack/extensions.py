import warnings

import torch

from .backpropextension import BackpropExtension
from .curvature import Curvature
from .secondorder.strategies import (BackpropStrategy, LossHessianStrategy,
                                     ExpectationApproximation)


class Extension():
    def __init__(self, savefield):
        self.savefield = savefield
        return

    def extension_to_trigger(self):
        return self.__class__


class VARIANCE(Extension):
    def __init__(self):
        super().__init__(savefield="variance")


class BATCH_L2(Extension):
    def __init__(self):
        super().__init__(savefield="batch_l2")


class BATCH_GRAD(Extension):
    def __init__(self):
        super().__init__(savefield="grad_batch")


class SUM_GRAD_SQUARED(Extension):
    def __init__(self):
        super().__init__(savefield="sum_grad_squared")


class DIAG_GGN(Extension):
    def __init__(self):
        super().__init__(savefield="diag_ggn")


class GRAD(Extension):
    def __init__(self):
        super().__init__(savefield="grad_bpext")


class DIAG_H(Extension):
    def __init__(self):
        super().__init__(savefield="diag_h")


class ParametrizedExtension(Extension):
    def __init__(self, savefield, input):
        self.input = input
        super().__init__(savefield=savefield)


class CMP(ParametrizedExtension):
    def __init__(self, curv_type):
        super().__init__(savefield="cmp", input=curv_type)

    def get_curv_type(self):
        return self.input


class HBP(ParametrizedExtension):
    def __init__(
            self,
            curv_type,
            loss_hessian_strategy,
            backprop_strategy,
            ea_strategy,
            savefield="hbp",
    ):
        super().__init__(
            savefield=savefield,
            input=[
                curv_type,
                loss_hessian_strategy,
                backprop_strategy,
                ea_strategy,
            ])

    def get_curv_type(self):
        return self.input[0]

    def get_loss_hessian_strategy(self):
        return self.input[1]

    def get_backprop_strategy(self):
        return self.input[2]

    def get_ea_strategy(self):
        return self.input[3]


class KFAC(HBP):
    def __init__(self):
        super().__init__(
            curv_type=Curvature.GGN,
            loss_hessian_strategy=LossHessianStrategy.SAMPLING,
            backprop_strategy=BackpropStrategy.SQRT,
            ea_strategy=ExpectationApproximation.BOTEV_MARTENS,
            savefield="kfac",
        )

    def extension_to_trigger(self):
        return HBP


class KFRA(HBP):
    def __init__(self):
        super().__init__(
            curv_type=Curvature.GGN,
            loss_hessian_strategy=LossHessianStrategy.AVERAGE,
            backprop_strategy=BackpropStrategy.BATCH_AVERAGE,
            ea_strategy=ExpectationApproximation.BOTEV_MARTENS,
            savefield="kfra",
        )

    def extension_to_trigger(self):
        return HBP


class KFLR(HBP):
    def __init__(self):
        super().__init__(
            curv_type=Curvature.GGN,
            loss_hessian_strategy=LossHessianStrategy.EXACT,
            backprop_strategy=BackpropStrategy.SQRT,
            ea_strategy=ExpectationApproximation.BOTEV_MARTENS,
            savefield="kflr",
        )

    def extension_to_trigger(self):
        return HBP


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
        HBP,
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
            key = (module.__class__, ext.extension_to_trigger())

            if key in Extensions.registeredExtensions:
                yield Extensions.registeredExtensions[key]
