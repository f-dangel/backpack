from ..context import CTX
from ..jacobians.mseloss import MSELossJacobian
from ..jacobians.crossentropyloss import CrossEntropyLossJacobian
from .diaghbase import DiagHBase


class DiagHLoss(DiagHBase):

    def backpropagate(self, module, grad_input, grad_output):
        sqrt_H = self.sqrt_hessian(module, grad_input, grad_output)
        CTX._backpropagated_sqrt_h = [sqrt_H]
        CTX._backpropagated_sqrt_h_signs = [1.]


class DiagHMSELoss(DiagHLoss, MSELossJacobian):
    pass


class DiagHCrossEntropyLoss(DiagHLoss, CrossEntropyLossJacobian):
    pass


EXTENSIONS = [
    DiagHMSELoss(),
    DiagHCrossEntropyLoss(),
]
