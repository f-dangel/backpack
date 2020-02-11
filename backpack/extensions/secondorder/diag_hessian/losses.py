from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule


class DiagHLoss(DiagHBaseModule):
    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        sqrt_H = self.derivatives.sqrt_hessian(module, g_inp, g_out)
        return {"matrices": [sqrt_H], "signs": [self.PLUS]}


class DiagHMSELoss(DiagHLoss):
    def __init__(self):
        super().__init__(derivatives=MSELossDerivatives())


class DiagHCrossEntropyLoss(DiagHLoss):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivatives())
