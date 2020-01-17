from .module_extension import ModuleExtension


class MatToJacMat(ModuleExtension):
    """
    Base class for backpropagating matrices by multiplying with Jacobians.
    """

    def __init__(self, derivatives, params=None):
        super().__init__(params)
        self.derivatives = derivatives

    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):

        # TODO Need confirmation: Is this an artifact from the Flatten layer?
        # If so: It is no longer a no-op, so it should be removed
        if self.derivatives is None:
            return backproped
        # end of TODO

        if isinstance(backproped, list):
            M_list = [
                self.derivatives.jac_t_mat_prod(module, grad_inp, grad_out, M)
                for M in backproped
            ]
            return list(M_list)
        else:
            return self.derivatives.jac_t_mat_prod(
                module, grad_inp, grad_out, backproped
            )
