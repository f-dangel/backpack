from ...ctxinteract import ActOnCTX
from ...backpropextension import BackpropExtension


class HBPBase(BackpropExtension, ActOnCTX):
    """Backpropagate batch averaged Hessians."""

    def __init__(self, ctx_name, extension, params=None):
        if params is None:
            params = []
        ActOnCTX.__init__(self, ctx_name)
        BackpropExtension.__init__(
            self, self.get_module(), extension, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        ea_h_out = self.get_from_ctx()
        ea_h_in = self.backpropagate_ggn_term(module, grad_input, grad_output,
                                              ea_h_out)
        ea_h_in = self.add_residual_term(module, grad_input, grad_output,
                                         ea_h_in)
        self.set_in_ctx(ea_h_in)

    def backpropagate_ggn_term(self, module, grad_input, grad_output,
                               ea_h_out):
        """Given EA of the output Hessian, compute EA of the input Hessian."""
        raise NotImplementedError

    def add_residual_term(module, grad_input, grad_output, ea_h_in):
        """Second-order effects introduced by the module function."""
        raise NotImplementedError
