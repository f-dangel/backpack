from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from .diag_h_base import DiagHBaseModule


class DiagHLoss(DiagHBaseModule):
    def backpropagate(self, ext, module, grad_input, grad_output, backproped):
        sqrt_H = self.derivatives.sqrt_hessian(module, grad_input, grad_output)
        return {
            "matrices": [sqrt_H],
            "signs": [self.PLUS]
        }


class DiagHMSELoss(DiagHLoss):
    def __init__(self):
        super().__init__(derivatives=MSELossDerivatives())


class DiagHCrossEntropyLoss(DiagHLoss):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivatives())
