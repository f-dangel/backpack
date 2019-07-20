import warnings

import torch

from .backpropextension import BackpropExtension

class Extension():
    def __init__(self, savefield):
        self.savefield = savefield
        return

    def extension_to_trigger(self):
        return self.__class__


class ParametrizedExtension(Extension):
    def __init__(self, savefield, input):
        self.input = input
        super().__init__(savefield=savefield)

class Extensions:
    EXTENSIONS = [
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
