from .matbackprop import ActOnCTX
from .backpropextension import BackpropExtension

class MatToJacMatJac(BackpropExtension, ActOnCTX):
    """Backprop `M` to `J^T M J`, where `M` is a batch expectation."""

    def __init__(self, ctx_name, extension, params=None):
        if params is None:
            params = []
        ActOnCTX.__init__(self, ctx_name)
        BackpropExtension.__init__(
            self, self.get_module(), extension, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        ea_h_out = self.get_from_ctx()
        ea_h_in = self.expectation_approximation(module, grad_input,
                                                 grad_output, ea_h_out)
        self.set_in_ctx(ea_h_in)

    def expectation_approximation(module, grad_input, grad_output, sqrt_out):
        """Given EA of the output Hessian, compute EA of the input Hessian."""
        raise NotImplementedError
