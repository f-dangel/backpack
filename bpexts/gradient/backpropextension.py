

class BackpropExtension():

    def __init__(self, module, extension, req_inputs=[], req_output=False):
        self.__module = module
        self.__extension = extension
        self.__req_inputs = req_inputs
        self.__req_output = req_output

    def apply(self, module, grad_input, grad_output):
        raise NotImplementedError
