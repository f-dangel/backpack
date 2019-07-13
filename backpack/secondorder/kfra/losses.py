from ...core.derivatives.mseloss import MSELossDerivatives
from ...core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from .kfrabase import KFRABase


class KFRALoss(KFRABase):
    def backpropagate(self, module, grad_input, grad_output):
        batch_averaged_hessian = self.sum_hessian(module, grad_input,
                                                  grad_output)
        self.set_mat_in_ctx(batch_averaged_hessian)


class KFRAMSELoss(KFRALoss, MSELossDerivatives):
    pass


class KFRACrossEntropyLoss(KFRALoss, CrossEntropyLossDerivatives):
    pass


EXTENSIONS = [KFRACrossEntropyLoss(), KFRAMSELoss()]
