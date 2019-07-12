from ...core.derivatives.mseloss import MSELossDerivatives
from ...core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from .diaghbase import DiagHBase


class DiagHLoss(DiagHBase):
    def backpropagate(self, module, grad_input, grad_output):
        sqrt_H = self.sqrt_hessian(module, grad_input, grad_output)

        self.set_mat_in_ctx([sqrt_H])
        self.set_sign_list_in_ctx([self.PLUS])


class DiagHMSELoss(DiagHLoss, MSELossDerivatives):
    pass


class DiagHCrossEntropyLoss(DiagHLoss, CrossEntropyLossDerivatives):
    pass


EXTENSIONS = [
    DiagHMSELoss(),
    DiagHCrossEntropyLoss(),
]
