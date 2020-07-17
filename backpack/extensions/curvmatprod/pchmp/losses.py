from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.extensions.curvmatprod.pchmp.pchmpbase import PCHMPBase


class PCHMPLoss(PCHMPBase):
    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        if not self.derivatives.hessian_is_psd():
            raise ValueError("Only convex losses supported.")

        def h_in_mat_prod(mat):
            """Multiplication with curvature matrix w.r.t. the module input.

            Parameters:
            -----------
            mat : torch.Tensor
                Matrix that will be multiplied.
            """
            return self.derivatives.make_hessian_mat_prod(module, g_inp, g_out)(mat)

        return h_in_mat_prod


class PCHMPMSELoss(PCHMPLoss):
    def __init__(self):
        super().__init__(derivatives=MSELossDerivatives())


class PCHMPCrossEntropyLoss(PCHMPLoss):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivatives())
