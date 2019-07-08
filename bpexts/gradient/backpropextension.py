import warnings


class BackpropExtension():

    def __init__(self, module, extension, params=None, req_inputs=None, req_output=False):
        if params is None:
            params = []
        if req_inputs is None:
            req_inputs = []

        # TODO: req_inputs and req_output are currently unused
        self.__module = module
        self.__extension = extension
        self.__req_inputs = req_inputs
        self.__req_output = req_output
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

    def __get_key(self):
        return tuple([self.__module, self.__extension])

    def __get_ext(self):
        return self.__extension

    def apply(self, module, grad_input, grad_output):
        for param in self.params:
            if (getattr(module, param) is not None) and (getattr(module, param).requires_grad):
                extFunc = getattr(self, param)
                extValue = extFunc(module, grad_input, grad_output)
                setattr(getattr(module, param), self.savefield, extValue)
        self.backpropagate(module, grad_input, grad_output)

    def backpropagate(self, module, grad_input, grad_output):
        warnings.warn("Backpropagate has not been overwritten")
