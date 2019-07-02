

class BackpropExtension():

    def __init__(self, req_inputs=[], req_outputs=[]):
        self.__req_inputs = req_inputs
        self.__req_outputs = req_outputs

    def apply(self, module, grad_input, grad_output):
        raise NotImplementedError
