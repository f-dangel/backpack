from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from .diag_ggn_base import DiagGGNBaseModule


class DiagGGNLoss(DiagGGNBaseModule):
    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        return self.derivatives.sqrt_hessian(
            module, grad_inp, grad_out, backproped
        )


class DiagGGNMSELoss(DiagGGNLoss):
    def __init__(self):
        super().__init__(derivatives=MSELossDerivatives())


class DiagGGNCrossEntropyLoss(DiagGGNLoss):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivatives())
