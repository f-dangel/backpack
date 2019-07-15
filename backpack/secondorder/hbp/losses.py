from ...core.derivatives.mseloss import MSELossDerivatives
from ...core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from .hbpbase import HBPBase


class HBPLoss(HBPBase):
    def backpropagate(self, module, grad_input, grad_output):
        batch_averaged_hessian = self.sum_hessian(module, grad_input,
                                                  grad_output)
        self.set_mat_in_ctx(batch_averaged_hessian)


class HBPMSELoss(HBPLoss, MSELossDerivatives):
    pass


class HBPCrossEntropyLoss(HBPLoss, CrossEntropyLossDerivatives):
    pass


EXTENSIONS = [HBPCrossEntropyLoss(), HBPMSELoss()]
