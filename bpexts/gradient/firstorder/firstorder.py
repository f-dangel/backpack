from ..backpropextension import BackpropExtension


class FirstOrderExtension(BackpropExtension):

    def __init__(self, module, extension, params=["bias", "weight"]):
        super().__init__(module, extension)
        self.params = params
        self.savefield = extension.savefield

        for param in self.params:
            extFunc = getattr(self, param, None)
            if extFunc is None:
                raise ValueError(
                    "Extension creation for " +
                    "[{},{}] ".format(module, extension) +
                    "failed: no function called {}".format(param)
                )

    def apply(self, module, grad_input, grad_output):
        for param in self.params:
            if (getattr(module, param) is not None) and (getattr(module, param).requires_grad):
                extFunc = getattr(self, param)
                extValue = extFunc(module, grad_input, grad_output)
                setattr(getattr(module, param), self.savefield, extValue)
