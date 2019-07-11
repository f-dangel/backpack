from ...context import CTX
from ...core.derivatives.mseloss import MSELossDerivatives
from ...core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from .diaghbase import DiagHBase


class DiagHLoss(DiagHBase):
    def backpropagate(self, module, grad_input, grad_output):
        sqrt_H = self.sqrt_hessian(module, grad_input, grad_output)
        CTX._backpropagated_sqrt_h = [sqrt_H]
        CTX._backpropagated_sqrt_h_signs = [1.]


class DiagHMSELoss(DiagHLoss, MSELossDerivatives):
    pass


class DiagHCrossEntropyLoss(DiagHLoss, CrossEntropyLossDerivatives):
    pass


EXTENSIONS = [
    DiagHMSELoss(),
    DiagHCrossEntropyLoss(),
]
