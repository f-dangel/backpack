from ...core.derivatives.mseloss import MSELossDerivatives
from ...core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from .kflrbase import KFLRBase


class KFLRLoss(KFLRBase):
    def backpropagate(self, module, grad_input, grad_output):
        sqrt_H = self.sqrt_hessian(module, grad_input, grad_output)
        self.set_mat_in_ctx(sqrt_H)


class KFLRMSELoss(KFLRLoss, MSELossDerivatives):
    pass


class KFLRCrossEntropyLoss(KFLRLoss, CrossEntropyLossDerivatives):
    pass


EXTENSIONS = [KFLRCrossEntropyLoss(), KFLRMSELoss()]
